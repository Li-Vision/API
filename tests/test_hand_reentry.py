"""
Entrada da mão no quadro: o app não envia frames sem mão, então o servidor só
percebe que a mão saiu pelo intervalo entre frames. A chegada da mão não pode
encher a janela do detector dinâmico (log real: OI com a mão aberta, sem gesto).
"""
from collections import deque

from src.api.holistic import HolisticFrame, MockLandmark
from src.data_collection import holistic_features as hf
from src.detectors.ml_detectors.lstm_model import GestureLSTM, LSTMGestureDetector

WINDOW = 15
FRAME_DT = 0.1  # ~10 fps


def _hand(dx=0.0):
    return [MockLandmark(0.5 + dx + i * 0.001, 0.5 + i * 0.001, 0.0) for i in range(21)]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _detector(clock, gap=0.5, skip=6):
    det = LSTMGestureDetector.__new__(LSTMGestureDetector)
    det.model = GestureLSTM(130, 8, 1, 2, dropout=0.0).eval()
    det.classes = ["X", "Y"]
    det.input_size = 130
    det.feature_schema = hf.SCHEMA_HANDS_V1
    det.window_size = WINDOW
    det.threshold = 0.0
    det.min_motion = 0.0
    det.buffer = deque(maxlen=WINDOW)
    det.clock = clock
    det.reentry_gap_seconds = gap
    det.reentry_skip_frames = skip
    return det


def _feed(det, clock, n, step=0.02):
    """Envia n frames seguidos; devolve quantos frames até a janela responder (ou None)."""
    for i in range(n):
        clock.now += FRAME_DT
        label, _ = det.detect(HolisticFrame(hands=[_hand(i * step)]))
        if label:
            return i + 1
    return None


def test_first_frames_of_a_session_are_skipped():
    clock = _Clock()
    det = _detector(clock, skip=6)
    # sem o descarte a janela responderia no frame 15; agora só no 21 (6 + 15)
    assert _feed(det, clock, 40) == WINDOW + 6


def test_hand_coming_back_after_a_pause_restarts_the_window():
    clock = _Clock()
    det = _detector(clock, skip=6)
    assert _feed(det, clock, 40) is not None  # sessão aquecida e respondendo

    clock.now += 2.0  # mão fora do quadro: o app parou de enviar
    # a janela só pode voltar a responder depois do descarte + 15 frames novos
    assert _feed(det, clock, 40) == WINDOW + 6


def test_short_gap_is_not_a_reentry():
    clock = _Clock()
    det = _detector(clock, gap=0.5, skip=6)
    _feed(det, clock, 40)

    clock.now += 0.3  # menor que o intervalo de reentrada: mesma mão, só um atraso de rede
    label, _ = det.detect(HolisticFrame(hands=[_hand(0.5)]))
    assert label in ("X", "Y")


def test_disabled_by_default_keeps_old_behavior():
    clock = _Clock()
    det = _detector(clock, gap=0.0, skip=0)
    assert _feed(det, clock, 40) == WINDOW

    clock.now += 5.0
    det.detect(HolisticFrame(hands=[_hand()]))
    assert len(det.buffer) == WINDOW  # sem reentrada configurada, o buffer não é limpo


def test_frame_without_hand_resets_the_entry():
    clock = _Clock()
    det = _detector(clock, skip=6)
    _feed(det, clock, 40)

    det.detect(HolisticFrame(hands=[]))  # frame explícito sem mão
    assert len(det.buffer) == 0
    assert _feed(det, clock, 40) == WINDOW + 6
