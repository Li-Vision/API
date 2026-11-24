import logging
from typing import Optional, Dict, Any, List
from src.core.supabase_client import supabase, supabase_admin
from src.core.pagination import fetch_all
from src.core.retry import retry_transient
from src.data_collection.static_collector import StaticCollector
from src.data_collection import holistic_features as hf
from src.api.holistic import HolisticFrame
from src.services.collection_queue import CollectJob, collection_queue
from src.validation import sample_validator as sv

logger = logging.getLogger(__name__)

# Frames por sequencia dinamica (janela do modelo).
DYNAMIC_WINDOW = 15

# nome -> (id, schema, guardado_em). Compartilhado por todas as sessoes: sem
# ele cada captura pagava 2 consultas ao Supabase so' para achar o dataset.
_DATASET_CACHE: Dict[str, tuple] = {}
_DATASET_CACHE_TTL = 120.0


def _forget_dataset(dataset_id: Optional[str] = None) -> None:
    for name, (ds_id, _, _) in list(_DATASET_CACHE.items()):
        if dataset_id is None or ds_id == dataset_id:
            _DATASET_CACHE.pop(name, None)
    collection_queue.invalidate(dataset_id)


class CollectionService:
    def __init__(self):
        self.static_collector = StaticCollector()
        self.dynamic_buffer = []
        self.dynamic_session: Optional[Dict[str, Any]] = None

    def _get_or_create_dataset(self, name: str, ds_type: str,
                               feature_schema: str = hf.DEFAULT_SCHEMA) -> str:
        import time
        name = name.upper()
        cached = _DATASET_CACHE.get(name)
        if cached and time.time() - cached[2] < _DATASET_CACHE_TTL:
            return cached[0]

        res = supabase.table("datasets").select("id").eq("name", name).execute()
        if len(res.data) > 0:
            ds_id = res.data[0]["id"]
        else:
            ins_res = supabase.table("datasets").insert({
                "name": name,
                "type": ds_type,
                "feature_schema": feature_schema,
            }).execute()
            ds_id = ins_res.data[0]["id"]
        _DATASET_CACHE[name] = (ds_id, None, time.time())
        return ds_id

    def _get_dataset_schema(self, dataset_id: str) -> Optional[str]:
        """Schema de um dataset existente, ou None se for anterior à coluna."""
        import time
        for name, (ds_id, schema, at) in _DATASET_CACHE.items():
            if ds_id == dataset_id and schema and time.time() - at < _DATASET_CACHE_TTL:
                return schema
        schema = self._read_dataset_schema(dataset_id)
        for name, (ds_id, _, at) in list(_DATASET_CACHE.items()):
            if ds_id == dataset_id:
                _DATASET_CACHE[name] = (ds_id, schema, at)
        return schema

    def _read_dataset_schema(self, dataset_id: str) -> Optional[str]:
        try:
            res = supabase.table("datasets").select("feature_schema").eq("id", dataset_id).execute()
            if res.data:
                return res.data[0].get("feature_schema")
        except Exception as exc:  # coluna pode não existir (migração pendente)
            logger.warning("[collect] leitura de feature_schema falhou: %s", exc)
        return None

    def collect_static(self, label: str, hand, dataset_name: str, user_id: str = None) -> dict:
        """
        Coleta uma amostra estática.

        `hand` pode ser a mão isolada (formato legado, 42 features) ou um
        HolisticFrame com pose/face. O schema é decidido na criação do dataset
        e, para datasets já existentes, o formato gravado antes é respeitado —
        assim amostras novas nunca entram com dimensão diferente das antigas.
        """
        try:
            dataset_name = dataset_name.upper()
            label = label.upper()

            frame = hand if isinstance(hand, HolisticFrame) else None
            wants_holistic = bool(frame and frame.is_holistic)

            dataset_id = self._get_or_create_dataset(
                dataset_name,
                "static",
                hf.SCHEMA_HOLISTIC_V1 if wants_holistic else hf.DEFAULT_SCHEMA,
            )

            # Dataset preexistente manda no formato: misturar dimensões dentro
            # do mesmo dataset inviabiliza o treino.
            existing_schema = self._get_dataset_schema(dataset_id)
            requested_schema = hf.SCHEMA_HOLISTIC_V1 if wants_holistic else hf.SCHEMA_HANDS_V1
            if existing_schema and existing_schema != requested_schema:
                return {
                    "ok": False,
                    "error": (
                        f"O dataset '{dataset_name}' e '{existing_schema}', mas a coleta "
                        f"atual e '{requested_schema}'. Troque o modo de coleta ou "
                        "escolha/crie outro dataset."
                    ),
                }
            use_holistic = wants_holistic

            hand_landmarks = frame.hands[0] if frame and frame.hands else hand
            if use_holistic:
                features = hf.build_frame_vector(frame, hf.SCHEMA_HOLISTIC_V1)
            else:
                features = self.static_collector.landmarks_to_features(hand_landmarks)

            # Integridade na hora (sem rede): amostra corrompida nem entra na fila.
            raw_issues = sv.check_hand_landmarks(hand_landmarks)
            issues = raw_issues + sv.check_integrity(features, sv.KIND_STATIC)
            if any(i.severity == "error" for i in issues):
                return self._rejected(label, dataset_id, issues)

            return self._enqueue(CollectJob(
                dataset_id=dataset_id, label=label, kind=sv.KIND_STATIC,
                features=features, user_id=user_id,
                # A borda da camera so' da' para ver nos landmarks crus.
                extra_issues=[i for i in raw_issues if i.severity != "error"],
            ), count_key="sample_count")
        except Exception as e:
            logger.error(f"[collect_static] Erro: {str(e)}")
            return {"ok": False, "error": f"Backend Error: {str(e)}"}

    def _rejected(self, label: str, dataset_id: str, issues) -> dict:
        """Resposta de amostra corrompida: `ok: False` faz o app mostrar o motivo."""
        errors = [i for i in issues if i.severity == "error"]
        logger.info("[collect] amostra rejeitada %s/%s: %s", dataset_id, label,
                    ", ".join(i.code for i in errors))
        return {
            "ok": False,
            "status": sv.STATUS_REJECTED,
            "error": "Amostra descartada: " + "; ".join(i.message for i in errors),
            "reasons": [i.to_dict() for i in issues],
        }

    def _enqueue(self, job: CollectJob, count_key: str) -> dict:
        """Manda para a fila de validacao e responde na hora com o ticket."""
        ticket = collection_queue.submit(job)
        if ticket.get("status") == "error":
            return {"ok": False, "error": ticket.get("error")}

        count = collection_queue.known_count(job.dataset_id, job.label)
        if count is None:
            # 1a captura do dataset neste processo: a referencia ainda nao foi
            # carregada pelo worker. Uma contagem simples mantem o contador do
            # app certo sem esperar a validacao.
            res = retry_transient(lambda: supabase.table("samples").select("id", count="exact")
                                  .eq("dataset_id", job.dataset_id).eq("label", job.label)
                                  .limit(1).execute())
            count = res.count if getattr(res, "count", None) is not None else len(res.data)

        return {
            "ok": True,
            "queued": True,
            "status": ticket["status"],
            "ticket_id": ticket["ticket_id"],
            # Aceitas ate' agora; a desta captura entra quando o ticket fechar.
            count_key: count,
        }

    def collect_dynamic_start(self, label: str, dataset_name: str, user_id: str = None) -> dict:
        # O dataset é criado de forma preguiçosa no primeiro frame, quando o
        # feature_schema (hands_v1 vs holistic_v1) já pode ser detectado.
        self.dynamic_session = {
            "label": label.upper(),
            "dataset_name": dataset_name.upper(),
            "user_id": user_id,
            "dataset_id": None,
            "feature_schema": None,
        }
        self.dynamic_buffer = []
        return {"ok": True, "status": "started"}

    def collect_dynamic_frame(self, frame) -> dict:
        """
        Recebe um HolisticFrame (ou lista de mãos legada) e extrai features no
        mesmo formato consumido pelo SequenceGestureDetector, via módulo unificado.

        O feature_schema é decidido no primeiro frame: holistic_v1 se houver
        pose/face, caso contrário hands_v1 (130 features/frame). O dataset é
        criado com esse schema.
        """
        if not self.dynamic_session:
            return {"ok": False, "error": "No dynamic session active"}

        frame = frame if isinstance(frame, HolisticFrame) else HolisticFrame(hands=frame)

        # Decide o schema e cria o dataset no primeiro frame
        if self.dynamic_session["dataset_id"] is None:
            schema = hf.SCHEMA_HOLISTIC_V1 if frame.is_holistic else hf.SCHEMA_HANDS_V1
            self.dynamic_session["feature_schema"] = schema
            self.dynamic_session["dataset_id"] = self._get_or_create_dataset(
                self.dynamic_session["dataset_name"], "dynamic", schema
            )

            # Dataset preexistente manda no formato: misturar "so maos" com
            # holistico no mesmo dataset gera sequencias de tamanhos diferentes
            # e inviabiliza o treino (mesma regra da coleta estatica).
            existing_schema = self._get_dataset_schema(self.dynamic_session["dataset_id"])
            if existing_schema and existing_schema != schema:
                self.dynamic_session = None
                self.dynamic_buffer = []
                return {
                    "ok": False,
                    "error": (
                        f"O dataset e '{existing_schema}', mas a coleta atual e '{schema}'. "
                        "Troque o modo (Holistico / So maos) ou escolha outro dataset."
                    ),
                }

        schema = self.dynamic_session["feature_schema"]
        features = hf.build_frame_vector(frame, schema)
        self.dynamic_buffer.append(features)

        if len(self.dynamic_buffer) >= DYNAMIC_WINDOW:
            return self.collect_dynamic_save()

        return {
            "ok": True,
            "collecting": True,
            "buffer": len(self.dynamic_buffer),
            "feature_schema": schema,
        }

    def collect_dynamic_save(self) -> dict:
        try:
            if not self.dynamic_session or len(self.dynamic_buffer) < DYNAMIC_WINDOW:
                return {"ok": False, "error": "Incomplete buffer or no session"}

            dataset_id = self.dynamic_session["dataset_id"]
            label = self.dynamic_session["label"]
            user_id = self.dynamic_session.get("user_id")

            # Achata os 15 frames num unico vetor (formato gravado em `samples`).
            flattened_features = []
            for frame in self.dynamic_buffer:
                flattened_features.extend(frame)
            self.dynamic_buffer = []

            issues = sv.check_integrity(flattened_features, sv.KIND_DYNAMIC, window=DYNAMIC_WINDOW)
            if any(i.severity == "error" for i in issues):
                return self._rejected(label, dataset_id, issues)

            return self._enqueue(CollectJob(
                dataset_id=dataset_id, label=label, kind=sv.KIND_DYNAMIC,
                features=flattened_features, user_id=user_id, window=DYNAMIC_WINDOW,
            ), count_key="sequences")
        except Exception as e:
            logger.error(f"[collect_dynamic_save] Erro: {str(e)}")
            return {"ok": False, "error": f"Backend Error: {str(e)}"}

    def collect_dynamic_stop(self) -> dict:
        self.dynamic_session = None
        self.dynamic_buffer = []
        return {"ok": True, "status": "stopped"}

    def get_datasets(self) -> dict:
        try:
            res = supabase.table("datasets").select("id, name, type, feature_schema").execute()
        except Exception as exc:  # coluna pode não existir (migração pendente)
            logger.warning("[collect] listagem com feature_schema falhou: %s", exc)
            res = supabase.table("datasets").select("id, name, type").execute()
        return {"datasets": res.data}

    def get_dataset_info(self, dataset_id: str) -> dict:
        ds_res = supabase.table("datasets").select("*").eq("id", dataset_id).execute()
        if len(ds_res.data) == 0:
            return {"error": "Dataset not found"}
            
        ds = ds_res.data[0]
        
        samples = fetch_all(lambda: supabase.table("samples").select("label").eq("dataset_id", dataset_id))
        
        label_counts = {}
        for row in samples:
            lbl = row["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            
        ds["stats"] = label_counts
        ds["total_samples"] = len(samples)
        
        return ds

    def delete_dataset(self, dataset_id: str, force: bool = False) -> dict:
        try:
            # Bloqueia a exclusão se houver modelos treinados vinculados a este
            # dataset — apagar o dataset quebraria a referência desses modelos.
            models_res = supabase.table("models").select("id, name, storage_path").eq("dataset_id", dataset_id).execute()
            if models_res.data and not force:
                model_names = ", ".join(m["name"] for m in models_res.data)
                return {
                    "ok": False,
                    "error": f"Existem modelos treinados vinculados a este dataset ({model_names}). Excluí-los junto com o dataset?",
                    "requires_force": True,
                    "linked_models": [m["name"] for m in models_res.data],
                }

            if models_res.data:
                # force=True: apaga os modelos vinculados (arquivo no storage + registro) antes do dataset.
                for m in models_res.data:
                    storage_path = m.get("storage_path")
                    if storage_path:
                        try:
                            supabase.storage.from_("models").remove([storage_path])
                        except Exception as storage_err:
                            logger.warning("[delete_dataset] Falha ao remover arquivo do storage '%s': %s", storage_path, storage_err)
                    supabase.table("models").delete().eq("id", m["id"]).execute()
                logger.info("[delete_dataset] %d modelo(s) vinculado(s) removido(s) em cascata.", len(models_res.data))

            # Deleta samples primeiro (redundante com CASCADE, mas garante limpeza)
            supabase.table("samples").delete().eq("dataset_id", dataset_id).execute()
            # Depois deleta o dataset
            res = supabase.table("datasets").delete().eq("id", dataset_id).execute()
            _forget_dataset(dataset_id)
            logger.info("[delete_dataset] Dataset %s deletado. Resultado: %s", dataset_id, res.data)
            return {"ok": True}
        except Exception as e:
            logger.error("[delete_dataset] Erro: %s", str(e))
            return {"ok": False, "error": str(e)}

    def delete_label(self, dataset_id: str, label: str) -> dict:
        try:
            target_label = label.upper()
            logger.info("[delete_label] Deletando label='%s' do dataset='%s'", target_label, dataset_id)
            
            # Verifica quantas samples existem antes
            check = supabase.table("samples").select("id", count="exact").eq("dataset_id", dataset_id).eq("label", target_label).execute()
            count_before = check.count if hasattr(check, 'count') and check.count is not None else len(check.data)
            logger.info("[delete_label] Encontradas %d amostras para deletar", count_before)
            
            if count_before == 0:
                return {"ok": True, "deleted": 0, "message": "Nenhuma amostra encontrada para este gesto."}
            
            res = supabase.table("samples").delete().eq("dataset_id", dataset_id).eq("label", target_label).execute()
            collection_queue.invalidate(dataset_id)
            deleted_count = len(res.data) if res.data else 0
            logger.info("[delete_label] Deletadas %d amostras", deleted_count)
            return {"ok": True, "deleted": deleted_count}
        except Exception as e:
            logger.error("[delete_label] Erro: %s", str(e))
            return {"ok": False, "error": str(e)}

    def rename_dataset(self, dataset_id: str, new_name: str) -> dict:
        try:
            supabase.table("datasets").update({"name": new_name.upper()}).eq("id", dataset_id).execute()
            _forget_dataset(dataset_id)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def rename_label(self, dataset_id: str, old_label: str, new_label: str) -> dict:
        try:
            supabase.table("samples").update({"label": new_label.upper()}).eq("dataset_id", dataset_id).eq("label", old_label.upper()).execute()
            collection_queue.invalidate(dataset_id)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Validacao assincrona / quarentena
    # ------------------------------------------------------------------
    def get_ticket(self, ticket_id: str) -> dict:
        ticket = collection_queue.get(ticket_id)
        if ticket is None:
            return {"ok": False, "error": "Ticket não encontrado (expirado ou de outro servidor)."}
        return {"ok": True, **ticket}

    def list_quarantine(self, dataset_id: Optional[str] = None, label: Optional[str] = None) -> dict:
        from src.services.collection_queue import QUARANTINE_TABLE
        try:
            def query():
                q = supabase.table(QUARANTINE_TABLE).select(
                    "id, dataset_id, label, kind, user_id, reasons, scores, created_at")
                if dataset_id:
                    q = q.eq("dataset_id", dataset_id)
                if label:
                    q = q.eq("label", label.upper())
                return q
            return {"ok": True, "items": fetch_all(query)}
        except Exception as e:
            return {"ok": False, "error": f"Quarentena indisponível (migração aplicada?): {e}"}

    def get_quarantine_item(self, item_id: str) -> dict:
        """Item completo, com as features (para desenhar o gesto na revisao)."""
        from src.services.collection_queue import QUARANTINE_TABLE
        try:
            res = supabase.table(QUARANTINE_TABLE).select("*").eq("id", item_id).execute()
            if not res.data:
                return {"ok": False, "error": "Item não encontrado na quarentena."}
            return {"ok": True, "item": res.data[0]}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def label_example(self, dataset_id: str, label: str, max_rows: int = 60) -> dict:
        """
        Uma amostra tipica do sinal (medoide das mais recentes), para o admin
        comparar com a amostra em revisao.
        """
        try:
            ds = supabase.table("datasets").select("type").eq("id", dataset_id).execute().data
            kind = sv.KIND_DYNAMIC if ds and ds[0]["type"] == "dynamic" else sv.KIND_STATIC
            rows = (supabase.table("samples").select("features")
                    .eq("dataset_id", dataset_id).eq("label", label.upper())
                    .order("created_at", desc=True).limit(max_rows).execute().data)
            if not rows:
                return {"ok": True, "features": None, "count": 0}
            idx = sv.medoid_index([r["features"] for r in rows], kind)
            return {
                "ok": True,
                "features": rows[idx]["features"] if idx is not None else None,
                "count": len(rows),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def approve_quarantine(self, item_id: str, label: Optional[str] = None) -> dict:
        """
        Admin confirma que a amostra e' boa: ela vai para `samples` (entra no
        treino). `label` corrige o rotulo — o caso comum de quem gravou um
        sinal com o gesto de outro.
        """
        from src.services.collection_queue import QUARANTINE_TABLE
        try:
            res = supabase.table(QUARANTINE_TABLE).select(
                "id, dataset_id, label, features, user_id").eq("id", item_id).execute()
            if not res.data:
                return {"ok": False, "error": "Item não encontrado na quarentena."}
            item = res.data[0]
            row = {k: item[k] for k in ("dataset_id", "label", "features") }
            new_label = (label or "").strip().upper()
            if new_label and new_label != item["label"]:
                logger.info("[approve_quarantine] %s: rotulo corrigido %s -> %s",
                            item_id, item["label"], new_label)
                row["label"] = new_label
            if item.get("user_id"):
                row["user_id"] = item["user_id"]
            # Insere ANTES de apagar: se a segunda etapa falhar, sobra uma copia
            # na quarentena, nunca a amostra perdida.
            retry_transient(lambda: supabase.table("samples").insert(row).execute())
            supabase.table(QUARANTINE_TABLE).delete().eq("id", item_id).execute()
            collection_queue.invalidate(item["dataset_id"])
            return {"ok": True, "label": row["label"]}
        except Exception as e:
            logger.error("[approve_quarantine] Erro: %s", e)
            return {"ok": False, "error": str(e)}

    def discard_quarantine(self, item_id: str) -> dict:
        from src.services.collection_queue import QUARANTINE_TABLE
        try:
            supabase.table(QUARANTINE_TABLE).delete().eq("id", item_id).execute()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_ranking(self) -> dict:
        try:
            rows = fetch_all(lambda: supabase.table("samples").select("user_id"))
            counts = {}
            for row in rows:
                uid = row.get("user_id")
                if uid:
                    counts[uid] = counts.get(uid, 0) + 1
            
            if not counts:
                return {"ranking": []}
                
            uids = list(counts.keys())
            prof_res = supabase.table("profiles").select("id, full_name").in_("id", uids).execute()
            
            profiles = {p["id"]: p.get("full_name", "User") for p in prof_res.data}
            
            ranking = []
            for uid, count in counts.items():
                ranking.append({
                    "user_id": uid,
                    "name": profiles.get(uid, "Unknown User"),
                    "samples": count
                })
                
            ranking.sort(key=lambda x: x["samples"], reverse=True)
            return {"ranking": ranking}
        except Exception as e:
            return {"error": str(e)}