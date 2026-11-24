"""
Coleta assincrona com validacao: a rota enfileira, um worker valida e grava.

Antes cada captura fazia, dentro da requisicao, 3-4 idas ao Supabase (achar o
dataset, ler o schema, inserir, contar) e gravava QUALQUER coisa em `samples`.
Agora a rota so' checa a integridade (instantaneo, sem rede), enfileira e
responde com um `ticket_id`. O worker:

1. carrega (uma vez, com cache) a referencia do dataset — as amostras ja'
   gravadas, agrupadas por sinal;
2. valida a amostra (integridade + semelhanca + modelo ativo);
3. grava conforme o veredito:
   - accepted -> `samples` (entra no treino);
   - suspect  -> `samples_quarantine` (fora do treino ate' um admin aprovar);
   - rejected -> nao grava; o motivo fica no ticket.

O resultado e' consultado por `get(ticket_id)` / `wait(ticket_id)` (REST e
WebSocket usam isso).
"""

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.validation import sample_validator as sv

logger = logging.getLogger(__name__)

QUARANTINE_TABLE = "samples_quarantine"


@dataclass
class CollectJob:
    dataset_id: str
    label: str
    kind: str                       # "static" | "dynamic"
    features: list
    user_id: Optional[str] = None
    window: Optional[int] = None
    extra_issues: List[sv.Issue] = field(default_factory=list)


class SupabasePersistence:
    """Acesso ao banco usado pelo worker (substituivel nos testes)."""

    def __init__(self):
        from src.core.supabase_client import supabase
        self.db = supabase

    def load_rows(self, dataset_id: str, kind: str) -> list:
        from src.core.pagination import fetch_all
        # Amostras dinamicas tem ~4 mil floats: pagina pequena (ver training_service).
        page = 100 if kind == sv.KIND_DYNAMIC else 1000
        return fetch_all(
            lambda: self.db.table("samples").select("label, features").eq("dataset_id", dataset_id),
            page_size=page,
        )

    def model_names(self, dataset_id: str) -> List[str]:
        try:
            res = self.db.table("models").select("name").eq("dataset_id", dataset_id).execute()
            return [r["name"] for r in res.data or []]
        except Exception:
            return []

    def insert_sample(self, row: dict) -> None:
        from src.core.retry import retry_transient
        retry_transient(lambda: self.db.table("samples").insert(row).execute())

    def insert_quarantine(self, row: dict) -> Optional[str]:
        from src.core.retry import retry_transient
        res = retry_transient(lambda: self.db.table(QUARANTINE_TABLE).insert(row).execute())
        return (res.data or [{}])[0].get("id")


class CollectionQueue:
    def __init__(self, persistence=None, cfg: Optional[sv.ValidationConfig] = None,
                 model_check_factory: Optional[Callable] = None,
                 max_pending: int = 500, ticket_ttl: float = 900.0,
                 reference_ttl: float = 900.0):
        self._persistence = persistence
        self.cfg = cfg or sv.ValidationConfig()
        self._model_check_factory = model_check_factory
        self._queue: "queue.Queue[tuple]" = queue.Queue(maxsize=max_pending)
        self._tickets: Dict[str, dict] = {}
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        self._refs: Dict[str, tuple] = {}     # dataset_id -> (DatasetReference, carregado_em, modelos)
        self._ticket_ttl = ticket_ttl
        self._reference_ttl = reference_ttl
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ API

    @property
    def persistence(self):
        if self._persistence is None:
            self._persistence = SupabasePersistence()
        return self._persistence

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._run, name="collection-queue", daemon=True)
            self._thread.start()

    def submit(self, job: CollectJob) -> dict:
        """Enfileira e devolve o ticket na hora. Sobe o worker se preciso."""
        self.start()
        ticket_id = uuid.uuid4().hex
        ticket = {
            "ticket_id": ticket_id,
            "status": "queued",
            "label": job.label,
            "dataset_id": job.dataset_id,
            "kind": job.kind,
            "created_at": time.time(),
        }
        with self._lock:
            self._prune()
            self._tickets[ticket_id] = ticket
            self._events[ticket_id] = threading.Event()
        try:
            self._queue.put_nowait((ticket_id, job))
        except queue.Full:
            self._finish(ticket_id, {"status": "error",
                                     "error": "Fila de validação cheia; tente de novo em instantes."})
        return dict(ticket)

    def get(self, ticket_id: str) -> Optional[dict]:
        with self._lock:
            t = self._tickets.get(ticket_id)
            return dict(t) if t else None

    def wait(self, ticket_id: str, timeout: float = 30.0) -> Optional[dict]:
        ev = self._events.get(ticket_id)
        if ev is not None:
            ev.wait(timeout)
        return self.get(ticket_id)

    def pending(self) -> int:
        return self._queue.qsize()

    def known_count(self, dataset_id: str, label: str) -> Optional[int]:
        """Amostras aceitas do sinal segundo o cache (None se ainda nao carregado)."""
        with self._lock:
            entry = self._refs.get(dataset_id)
        return entry[0].counts.get(label, 0) if entry else None

    def invalidate(self, dataset_id: Optional[str] = None) -> None:
        """Descarta a referencia em cache (dataset alterado fora do worker)."""
        with self._lock:
            if dataset_id is None:
                self._refs.clear()
            else:
                self._refs.pop(dataset_id, None)

    def process_now(self, job: CollectJob) -> dict:
        """Processa sincronamente (testes e scripts). Devolve o ticket final."""
        ticket = self.submit_inline(job)
        self._process(ticket["ticket_id"], job)
        return self.get(ticket["ticket_id"])

    def submit_inline(self, job: CollectJob) -> dict:
        ticket_id = uuid.uuid4().hex
        with self._lock:
            self._tickets[ticket_id] = {"ticket_id": ticket_id, "status": "queued",
                                        "label": job.label, "dataset_id": job.dataset_id,
                                        "kind": job.kind, "created_at": time.time()}
            self._events[ticket_id] = threading.Event()
        return self.get(ticket_id)

    # --------------------------------------------------------------- worker

    def _run(self) -> None:
        while True:
            ticket_id, job = self._queue.get()
            try:
                self._process(ticket_id, job)
            except Exception as exc:  # o worker nunca morre por uma amostra
                logger.exception("[collect-queue] falha processando ticket %s", ticket_id)
                self._finish(ticket_id, {"status": "error", "error": f"Erro interno: {exc}"})
            finally:
                self._queue.task_done()

    def _reference(self, dataset_id: str, kind: str, window: Optional[int]):
        with self._lock:
            entry = self._refs.get(dataset_id)
        if entry and time.time() - entry[1] < self._reference_ttl and entry[0].kind == kind:
            return entry[0], entry[2]
        t0 = time.time()
        rows = self.persistence.load_rows(dataset_id, kind)
        ref = sv.DatasetReference.from_rows(rows, kind, window, self.cfg)
        models = self.persistence.model_names(dataset_id)
        with self._lock:
            self._refs[dataset_id] = (ref, time.time(), models)
        logger.info("[collect-queue] referencia do dataset %s: %d amostras em %.1fs",
                    dataset_id, len(rows), time.time() - t0)
        return ref, models

    def _model_check(self, kind: str, models: List[str]):
        if self._model_check_factory is not None:
            return self._model_check_factory(kind, models)
        try:
            from src.validation.model_check import build_model_check
            return build_model_check(kind, models)
        except Exception as exc:
            logger.warning("[collect-queue] checagem de modelo indisponivel: %s", exc)
            return None

    def _process(self, ticket_id: str, job: CollectJob) -> None:
        self._update(ticket_id, {"status": "processing"})
        t0 = time.time()

        ref, models = self._reference(job.dataset_id, job.kind, job.window)
        verdict = sv.validate(
            job.features, job.label, job.kind,
            reference=ref,
            model_check=self._model_check(job.kind, models),
            cfg=self.cfg,
            window=job.window,
            extra_issues=job.extra_issues,
        )

        row = {"dataset_id": job.dataset_id, "label": job.label, "features": job.features}
        if job.user_id:
            row["user_id"] = job.user_id

        result = {**verdict.to_dict(), "stored_in": None}
        if verdict.status == sv.STATUS_ACCEPTED:
            self.persistence.insert_sample(row)
            with self._lock:
                ref.add(job.label, job.features)
            result["stored_in"] = "samples"
        elif verdict.status == sv.STATUS_SUSPECT:
            try:
                qid = self.persistence.insert_quarantine({
                    **row,
                    "kind": job.kind,
                    "reasons": [i.to_dict() for i in verdict.issues],
                    "scores": verdict.scores,
                })
                result["stored_in"] = "quarantine"
                result["quarantine_id"] = qid
            except Exception as exc:
                # Tabela ausente (migracao pendente): NAO cai para `samples` —
                # o objetivo e' justamente manter a amostra fora do treino.
                logger.warning("[collect-queue] quarentena indisponivel (%s); amostra suspeita descartada", exc)
                result["quarantine_error"] = str(exc)

        result["sample_count"] = ref.counts.get(job.label, 0)
        result["ms"] = int((time.time() - t0) * 1000)
        logger.info("[collect-queue] %s %s/%s -> %s (%s) em %dms",
                    job.kind, job.dataset_id, job.label, verdict.status,
                    ", ".join(i.code for i in verdict.issues if i.severity != "info") or "ok",
                    result["ms"])
        self._finish(ticket_id, result)

    # -------------------------------------------------------------- tickets

    def _update(self, ticket_id: str, updates: dict) -> None:
        with self._lock:
            if ticket_id in self._tickets:
                self._tickets[ticket_id].update(updates)

    def _finish(self, ticket_id: str, updates: dict) -> None:
        self._update(ticket_id, {**updates, "finished_at": time.time()})
        ev = self._events.get(ticket_id)
        if ev is not None:
            ev.set()

    def _prune(self) -> None:
        """Esquece tickets antigos (chamado com o lock)."""
        limit = time.time() - self._ticket_ttl
        old = [k for k, t in self._tickets.items() if t.get("finished_at", time.time()) < limit]
        for k in old:
            self._tickets.pop(k, None)
            self._events.pop(k, None)


# Instancia unica do processo: REST e WebSocket compartilham fila e cache.
collection_queue = CollectionQueue()
