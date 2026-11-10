"""
Classificador estático pela forma da mão (`hand_shape_v1`) e filtro da
"segunda mão" duplicada.

Cobre o que fazia A/B serem trocados no app: o modelo antigo usava posição na
tela, corpo e rosto, e a medida de acurácia misturava quadros vizinhos entre
treino e teste.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_fake = types.ModuleType("src.core.supabase_client")
_fake.supabase = None
_fake.supabase_admin = None
sys.modules.setdefault("src.core.supabase_client", _fake)

from src.api.holistic import dedupe_hands, parse_input  # noqa: E402
from src.data_collection import hand_shape_features as hs  # noqa: E402
from src.data_collection import holistic_features as hf  # noqa: E402
from src.training import static_trainer as st  # noqa: E402


# ---------------------------------------------------------------------------
# Mãos sintéticas: duas "letras" com formas diferentes
# ---------------------------------------------------------------------------

def _shape(kind: str) -> np.ndarray:
    """21 pontos (x, y, z) em unidades de imagem; 'open' = dedos esticados."""
    pts = np.zeros((21, 3))
    for finger in range(5):
        base_x = (finger - 2) * 0.02
        for joint in range(4):
            idx = 1 + finger * 4 + joint
            reach = (joint + 1) * 0.03 if kind == "open" else 0.03 + (joint % 2) * 0.01
            pts[idx] = [base_x, -reach, 0.0]
    return pts


def _hand_dicts(kind, cx=0.5, cy=0.5, scale=1.0, noise=0.0, rng=None):
    pts = _shape(kind) * scale
    if noise and rng is not None:
        pts = pts + rng.normal(0, noise, pts.shape)
    return [{"x": float(cx + p[0]), "y": float(cy + p[1]), "z": float(p[2])} for p in pts]


def _holistic_row(kind, label, rng, user="u1", t=0.0, cx=0.5, cy=0.5):
    frame = parse_input({
        "hands": [_hand_dicts(kind, cx, cy, noise=0.002, rng=rng)],
        "pose": [{"x": 0.4, "y": 0.6, "z": 0.0, "visibility": 0.9} for _ in range(33)],
    })
    return {
        "label": label,
        "features": hf.build_frame_vector(frame, hf.SCHEMA_HOLISTIC_V1),
        "user_id": user,
        "created_at": t,
    }


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_vetor_tem_o_tamanho_declarado():
    frame = parse_input([_hand_dicts("open")])
    assert hs.hand_features(frame.hands[0]).shape == (hs.FEATURES,)


def test_forma_nao_depende_de_posicao_nem_de_distancia():
    perto = parse_input([_hand_dicts("open", cx=0.2, cy=0.3, scale=1.0)]).hands[0]
    longe = parse_input([_hand_dicts("open", cx=0.8, cy=0.7, scale=0.4)]).hands[0]
    np.testing.assert_allclose(hs.hand_features(perto), hs.hand_features(longe), atol=1e-9)


def test_formas_diferentes_dao_vetores_diferentes():
    aberta = parse_input([_hand_dicts("open")]).hands[0]
    fechada = parse_input([_hand_dicts("fist")]).hands[0]
    assert np.abs(hs.hand_features(aberta) - hs.hand_features(fechada)).max() > 0.1


def test_amostra_gravada_vira_o_mesmo_vetor_da_inferencia():
    """A conversão do dataset antigo tem de bater com o caminho ao vivo."""
    frame = parse_input({"hands": [_hand_dicts("open", cx=0.3, cy=0.6)],
                         "pose": [{"x": 0.4, "y": 0.6, "z": 0.0} for _ in range(33)]})
    stored = hf.build_frame_vector(frame, hf.SCHEMA_HOLISTIC_V1)

    from_dataset = hs.features_from_points(hs.points_from_holistic_vector(stored))
    live = hs.hand_features(frame.hands[0])
    np.testing.assert_allclose(from_dataset, live, atol=1e-9)


def test_bloco_de_mao_zerado_nao_vira_amostra():
    assert hs.points_from_holistic_vector([0.0] * 269) is None


def test_aumento_multiplica_e_mantem_rotulos():
    pts = np.stack([_shape("open"), _shape("fist")])
    out, labels = hs.augment_points(pts, ["A", "B"], np.random.default_rng(0), rotations=2)
    assert out.shape == (2 * 3 * 2, 21, 3)
    assert list(labels).count("A") == 6 and list(labels).count("B") == 6


# ---------------------------------------------------------------------------
# Segunda mão duplicada
# ---------------------------------------------------------------------------

def test_mao_detectada_duas_vezes_vira_uma():
    rng = np.random.default_rng(1)
    hands = parse_input([
        _hand_dicts("open", cx=0.50, cy=0.50),
        _hand_dicts("open", cx=0.51, cy=0.50, noise=0.003, rng=rng),
    ]).hands
    assert len(hands) == 1


def test_duas_maos_de_verdade_continuam_duas():
    hands = parse_input([
        _hand_dicts("open", cx=0.25, cy=0.5),
        _hand_dicts("fist", cx=0.75, cy=0.5),
    ]).hands
    assert len(hands) == 2


def test_dedupe_mantem_a_primeira():
    a = parse_input([_hand_dicts("open", cx=0.5)]).hands[0]
    b = parse_input([_hand_dicts("open", cx=0.505)]).hands[0]
    assert dedupe_hands([a, b]) == [a]


def test_aspect_e_lido_do_payload_e_valores_absurdos_sao_ignorados():
    assert parse_input({"hands": [_hand_dicts("open")], "aspect": 0.5625}).aspect == 0.5625
    assert parse_input({"hands": [_hand_dicts("open")], "aspect": 0}).aspect is None
    assert parse_input({"hands": [_hand_dicts("open")], "aspect": "x"}).aspect is None
    assert parse_input([_hand_dicts("open")]).aspect is None


# ---------------------------------------------------------------------------
# Sessões e holdout
# ---------------------------------------------------------------------------

def test_sessoes_separam_por_pessoa_letra_e_pausa():
    labels = ["A", "A", "A", "A", "B"]
    users = ["u", "u", "u", "v", "u"]
    times = [0, 5, 100, 6, 7]
    g = st.session_groups(labels, users, times)
    assert g[0] == g[1]          # mesma pessoa e letra, 5 s depois
    assert g[2] != g[0]          # pausa de 95 s: outra sessão
    assert g[3] != g[0]          # outra pessoa
    assert g[4] != g[0]          # outra letra


def test_holdout_nao_divide_a_mesma_sessao_quando_ha_varias():
    labels = np.array(["A"] * 100)
    groups = np.repeat(np.arange(5), 20)
    times = np.arange(100, dtype=float)
    train, test = st.session_holdout(labels, groups, times, test_fraction=0.2)

    assert len(test) == 20
    assert set(groups[test]).isdisjoint(set(groups[train]))
    assert set(groups[test]) == {4}  # a sessão mais recente


def test_holdout_de_sessao_unica_corta_o_trecho_final():
    labels = np.array(["A"] * 50)
    train, test = st.session_holdout(labels, np.zeros(50, int), np.arange(50, dtype=float))
    assert len(test) == 10
    assert min(test) > max(train)


# ---------------------------------------------------------------------------
# Treino + inferência
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(0)
    rows = []
    # Tudo gravado num canto da tela, por uma pessoa, em duas sessões por letra.
    for label, kind in (("A", "fist"), ("B", "open")):
        for session in range(2):
            for i in range(30):
                rows.append(_holistic_row(kind, label, rng, t=session * 1000 + i,
                                          cx=0.3, cy=0.7))
    return st.train_hand_shape_classifier(rows)


def test_treino_marca_o_formato_no_modelo(trained):
    assert trained["model"].feature_schema_ == hs.SCHEMA_HAND_SHAPE_V1
    assert trained["classes"] == ["A", "B"]
    assert trained["evaluated"] is True
    assert trained["accuracy"] == pytest.approx(1.0)


def test_treino_recusa_dataset_de_uma_letra_so():
    rng = np.random.default_rng(0)
    rows = [_holistic_row("open", "A", rng, t=i) for i in range(10)]
    with pytest.raises(ValueError):
        st.train_hand_shape_classifier(rows)


def _detector(model, threshold=0.7):
    from src.detectors.ml_detectors.static_detector import MLDetector

    det = object.__new__(MLDetector)
    det.model = types.SimpleNamespace(model=model)
    det.threshold = threshold
    return det


def test_reconhece_em_outro_lugar_da_tela_e_sem_pose(trained):
    """O que o modelo antigo não fazia: mesma letra, outra posição, sem corpo."""
    det = _detector(trained["model"])
    assert det.uses_hand_shape and det.expects_holistic_frame

    so_maos = parse_input([_hand_dicts("open", cx=0.8, cy=0.2, scale=0.6)])
    assert det.detect(so_maos)[0] == "B"

    com_rosto = parse_input({
        "hands": [_hand_dicts("fist", cx=0.6, cy=0.4, scale=1.5)],
        "face": [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(478)],
    })
    assert det.detect(com_rosto)[0] == "A"


def test_mao_esquerda_espelhada_da_a_mesma_letra(trained):
    det = _detector(trained["model"])
    espelhada = [{"x": 1.0 - p["x"], "y": p["y"], "z": p["z"]} for p in _hand_dicts("open")]
    assert det.detect(parse_input([espelhada]))[0] == "B"


def test_modelo_de_forma_passa_pelo_manager(trained):
    from src.recognition.detector_manager import DetectorManager

    mgr = DetectorManager([_detector(trained["model"])], min_score=0.7, stability_frames=1)
    label, score = mgr.detect(parse_input([_hand_dicts("fist")]))
    assert label == "A" and score >= 0.7
