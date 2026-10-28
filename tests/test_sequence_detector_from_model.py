"""
SequenceGestureDetector.from_model: o detector dinâmico legado (.joblib) vinha do
ModelCache montado sem `features_per_frame`, e todo frame falhava com AttributeError
(sinais dinâmicos como "Oi" e "Obrigado" nunca eram reconhecidos no servidor).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.holistic import HolisticFrame  # noqa: E402
from src.api.holistic import MockLandmark  # noqa: E402
from src.detectors.ml_detectors.sequence_detector import SequenceGestureDetector  # noqa: E402

WINDOW = 4


class _FakeModel:
    """Modelo sklearn mínimo: 130 features por frame (hands_v1) × janela."""
    n_features_in_ = 130 * WINDOW
    classes_ = np.array(["OI", "OBRIGADO"])

    def predict_proba(self, X):
        assert X.shape == (1, 130 * WINDOW)
        return np.array([[0.1, 0.9]])


def _frame():
    hand = [MockLandmark(x=0.5 + i * 0.001, y=0.5, z=0.0) for i in range(21)]
    return HolisticFrame(hands=[hand])


def test_from_model_define_os_atributos_usados_na_inferencia():
    det = SequenceGestureDetector.from_model(_FakeModel(), WINDOW, 0.7)
    assert det.features_per_frame == 130
    assert det.expected_features == 130 * WINDOW
    assert det.feature_schema == "hands_v1"
    assert det.buffer.maxlen == WINDOW


def test_detect_reconhece_gesto_depois_de_encher_a_janela():
    det = SequenceGestureDetector.from_model(_FakeModel(), WINDOW, 0.7)
    for _ in range(WINDOW - 1):
        assert det.detect(_frame()) == (None, 0.0)
    label, score = det.detect(_frame())
    assert label == "OBRIGADO" and score > 0.7


def test_modelo_de_uma_classe_so_nao_e_usavel():
    class _OneClass(_FakeModel):
        classes_ = np.array(["OI"])

    assert SequenceGestureDetector.is_discriminative(_FakeModel())
    assert not SequenceGestureDetector.is_discriminative(_OneClass())
