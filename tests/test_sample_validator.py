"""
Validacao de amostras: corrompida e' rejeitada, fora do padrao vai para a
quarentena, boa entra no dataset. Tudo offline (Supabase stubado).
"""
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_fake = types.ModuleType("src.core.supabase_client")
_fake.supabase = None
_fake.supabase_admin = None
sys.modules.setdefault("src.core.supabase_client", _fake)

from src.api.holistic import parse_input  # noqa: E402
from src.data_collection import holistic_features as hf  # noqa: E402
from src.services.collection_queue import CollectJob, CollectionQueue  # noqa: E402
from src.validation import sample_validator as sv  # noqa: E402

RNG = np.random.default_rng(0)

# Mao aberta (B) e mao fechada (A), em coordenadas de imagem.
_OPEN = np.array([[0, 0], [-.08, -.05], [-.13, -.1], [-.16, -.15], [-.18, -.2],
                  [-.06, -.2], [-.07, -.3], [-.07, -.37], [-.07, -.43],
                  [0, -.21], [0, -.32], [0, -.4], [0, -.47],
                  [.05, -.2], [.06, -.3], [.06, -.37], [.06, -.42],
                  [.1, -.17], [.12, -.25], [.13, -.3], [.14, -.34]])
_FIST = _OPEN.copy()
_FIST[[6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20], 1] *= 0.35


def _hand(shape, wrist=(0.5, 0.7), noise=0.004):
    pts = shape + np.array(wrist) + RNG.normal(0, noise, shape.shape)
    return [{"x": float(x), "y": float(y), "z": float(RNG.normal(0, 0.01))} for x, y in pts]


def _holistic_vec(shape, **kw):
    return hf.build_frame_vector(parse_input({"hands": [_hand(shape, **kw)]}), hf.SCHEMA_HOLISTIC_V1)


def _rows(n=30):
    return ([{"label": "B", "features": _holistic_vec(_OPEN)} for _ in range(n)]
            + [{"label": "A", "features": _holistic_vec(_FIST)} for _ in range(n)])


# ---------------------------------------------------------------- integridade

def test_amostra_boa_passa_na_integridade():
    assert not [i for i in sv.check_integrity(_holistic_vec(_OPEN), sv.KIND_STATIC) if i.severity == "error"]


def test_nan_e_rejeitado():
    vec = _holistic_vec(_OPEN)
    vec[10] = float("nan")
    assert sv.validate(vec, "B", sv.KIND_STATIC).status == sv.STATUS_REJECTED


def test_tamanho_errado_e_rejeitado():
    assert sv.validate([0.1] * 100, "B", sv.KIND_STATIC).status == sv.STATUS_REJECTED


def test_mao_colapsada_e_rejeitada():
    vec = hf.build_frame_vector(parse_input({"hands": [[{"x": .5, "y": .5, "z": 0}] * 21]}),
                                hf.SCHEMA_HOLISTIC_V1)
    v = sv.validate(vec, "B", sv.KIND_STATIC)
    assert v.status == sv.STATUS_REJECTED
    assert any(i.code == "hand_collapsed" for i in v.issues)


def test_sequencia_congelada_e_rejeitada():
    frame = _holistic_vec(_OPEN)
    v = sv.validate(frame * 15, "OI", sv.KIND_DYNAMIC, window=15)
    assert v.status == sv.STATUS_REJECTED
    assert any(i.code == "frozen" for i in v.issues)


def test_sequencia_com_movimento_passa():
    seq = []
    for t in range(15):
        seq += _holistic_vec(_OPEN, wrist=(0.4 + t * 0.01, 0.7))
    v = sv.validate(seq, "OI", sv.KIND_DYNAMIC, window=15)
    assert v.status == sv.STATUS_ACCEPTED


def test_salto_de_rastreamento_e_suspeito():
    seq = []
    for t in range(15):
        seq += _holistic_vec(_OPEN, wrist=(0.2 if t < 7 else 0.8, 0.7))
    v = sv.validate(seq, "OI", sv.KIND_DYNAMIC, window=15)
    assert any(i.code == "tracking_jump" for i in v.issues)
    assert v.status == sv.STATUS_SUSPECT


def test_formato_legado_42():
    from src.data_collection.static_collector import StaticCollector
    hand = parse_input([_hand(_OPEN)]).hands[0]
    vec = StaticCollector().landmarks_to_features(hand)
    assert sv.validate(vec, "B", sv.KIND_STATIC).status == sv.STATUS_ACCEPTED


# ---------------------------------------------------------------- semelhanca

def test_amostra_do_sinal_certo_e_aceita():
    ref = sv.DatasetReference.from_rows(_rows(), sv.KIND_STATIC)
    v = sv.validate(_holistic_vec(_OPEN), "B", sv.KIND_STATIC, reference=ref)
    assert v.status == sv.STATUS_ACCEPTED, v.to_dict()


def test_mao_esquerda_espelhada_nao_vira_suspeita():
    ref = sv.DatasetReference.from_rows(_rows(), sv.KIND_STATIC)
    mirrored = _OPEN * np.array([-1, 1])
    v = sv.validate(_holistic_vec(mirrored), "B", sv.KIND_STATIC, reference=ref)
    assert v.status == sv.STATUS_ACCEPTED, v.to_dict()


def test_rotulo_trocado_vai_para_quarentena():
    ref = sv.DatasetReference.from_rows(_rows(), sv.KIND_STATIC)
    v = sv.validate(_holistic_vec(_FIST), "B", sv.KIND_STATIC, reference=ref)
    assert v.status == sv.STATUS_SUSPECT
    codes = {i.code for i in v.issues}
    assert "looks_like_other" in codes or "outlier" in codes


def test_lixo_ja_gravado_nao_contamina_a_referencia():
    rows = _rows()
    rows += [{"label": "B", "features": [float("nan")] * sv.HOLISTIC_FRAME} for _ in range(5)]
    ref = sv.DatasetReference.from_rows(rows, sv.KIND_STATIC)
    assert ref.reference_size("B") == 30
    assert ref.counts["B"] == 35


def test_sem_referencia_suficiente_so_checa_integridade():
    ref = sv.DatasetReference.from_rows(_rows(3), sv.KIND_STATIC)
    v = sv.validate(_holistic_vec(_FIST), "B", sv.KIND_STATIC, reference=ref)
    assert v.status == sv.STATUS_ACCEPTED
    assert any(i.code == "few_reference" for i in v.issues)


# ---------------------------------------------------------------- modelo

def _model_says(predicted, conf):
    return lambda features, label: {"model": "M", "predicted": predicted, "confidence": conf,
                                    "label_prob": 1 - conf, "knows_label": True}


def test_modelo_discordando_sozinho_nao_barra():
    """Amostra parecida com o dataset em que o modelo erra: e' a mais util para treinar."""
    ref = sv.DatasetReference.from_rows(_rows(), sv.KIND_STATIC)
    v = sv.validate(_holistic_vec(_OPEN), "B", sv.KIND_STATIC, reference=ref,
                    model_check=_model_says("A", 0.99))
    assert v.status == sv.STATUS_ACCEPTED
    assert any(i.code == "model_disagrees" and i.severity == "info" for i in v.issues)


def test_modelo_com_erro_nao_derruba_a_validacao():
    def broken(features, label):
        raise RuntimeError("dimensao")
    v = sv.validate(_holistic_vec(_OPEN), "B", sv.KIND_STATIC, model_check=broken)
    assert v.status == sv.STATUS_ACCEPTED


# ---------------------------------------------------------------- fila

class _FakeDB:
    def __init__(self, rows, quarantine_ok=True):
        self.rows = rows
        self.samples, self.quarantine = [], []
        self.quarantine_ok = quarantine_ok

    def load_rows(self, dataset_id, kind):
        return list(self.rows)

    def model_names(self, dataset_id):
        return []

    def insert_sample(self, row):
        self.samples.append(row)

    def insert_quarantine(self, row):
        if not self.quarantine_ok:
            raise RuntimeError("relation samples_quarantine does not exist")
        self.quarantine.append(row)
        return "q1"


def _queue(db):
    return CollectionQueue(persistence=db, model_check_factory=lambda kind, models: None)


def test_fila_grava_boa_em_samples():
    db = _FakeDB(_rows())
    t = _queue(db).process_now(CollectJob("ds", "B", sv.KIND_STATIC, _holistic_vec(_OPEN)))
    assert t["status"] == sv.STATUS_ACCEPTED and t["stored_in"] == "samples"
    assert len(db.samples) == 1 and t["sample_count"] == 31


def test_fila_manda_suspeita_para_quarentena():
    db = _FakeDB(_rows())
    t = _queue(db).process_now(CollectJob("ds", "B", sv.KIND_STATIC, _holistic_vec(_FIST)))
    assert t["status"] == sv.STATUS_SUSPECT and t["stored_in"] == "quarantine"
    assert not db.samples and len(db.quarantine) == 1
    assert db.quarantine[0]["reasons"]


def test_sem_tabela_de_quarentena_suspeita_nao_cai_em_samples():
    db = _FakeDB(_rows(), quarantine_ok=False)
    t = _queue(db).process_now(CollectJob("ds", "B", sv.KIND_STATIC, _holistic_vec(_FIST)))
    assert t["stored_in"] is None and not db.samples


def test_fila_em_background_responde_ticket_e_conclui():
    db = _FakeDB(_rows())
    q = _queue(db)
    ticket = q.submit(CollectJob("ds", "B", sv.KIND_STATIC, _holistic_vec(_OPEN)))
    assert ticket["status"] == "queued"
    done = q.wait(ticket["ticket_id"], timeout=10)
    assert done["status"] == sv.STATUS_ACCEPTED


def test_exemplo_do_sinal_ignora_corrompida_e_destoante():
    feats = [_holistic_vec(_OPEN) for _ in range(10)]
    feats.insert(0, [float("nan")] * sv.HOLISTIC_FRAME)
    feats.insert(1, _holistic_vec(_FIST))
    idx = sv.medoid_index(feats, sv.KIND_STATIC)
    assert idx not in (0, 1, None)


def test_auditoria_marca_as_ruins():
    sys.path.insert(0, str(ROOT / "scripts"))
    from audit_dataset import audit_rows

    rows = [{"id": i, **r} for i, r in enumerate(_rows())]
    rows.append({"id": "bad", "label": "B", "features": [0.0] * sv.HOLISTIC_FRAME})
    rows.append({"id": "swap", "label": "B", "features": _holistic_vec(_FIST)})
    by_id = {r["id"]: r["status"] for r in audit_rows(rows, sv.KIND_STATIC, sv.ValidationConfig())}
    assert by_id["bad"] == sv.STATUS_REJECTED
    assert by_id["swap"] == sv.STATUS_SUSPECT
    assert sum(s == sv.STATUS_ACCEPTED for s in by_id.values()) >= 55
