from collections import deque

import torch

from src.api.holistic import HolisticFrame, MockLandmark
from src.data_collection import holistic_features as hf
from src.detectors.ml_detectors.lstm_model import GestureLSTM, LSTMGestureDetector

WINDOW = 15


def _hand(dx=0.0):
    return [MockLandmark(0.5 + dx + i * 0.001, 0.5 + i * 0.001, 0.0) for i in range(21)]


def _detector(min_motion):
    det = LSTMGestureDetector.__new__(LSTMGestureDetector)
    det.model = GestureLSTM(130, 8, 1, 2, dropout=0.0).eval()
    det.classes = ["X", "Y"]
    det.input_size = 130
    det.feature_schema = hf.SCHEMA_HANDS_V1
    det.window_size = WINDOW
    det.threshold = 0.0
    det.min_motion = min_motion
    det.buffer = deque(maxlen=WINDOW)
    return det


def _run(det, step):
    out = (None, 0.0)
    for i in range(WINDOW):
        out = det.detect(HolisticFrame(hands=[_hand(i * step)]))
    return out


def test_window_motion_is_zero_for_still_hand():
    vecs = [hf.build_frame_vector(HolisticFrame(hands=[_hand()]), hf.SCHEMA_HANDS_V1)] * 5
    assert hf.window_motion(vecs) == 0.0


def test_gate_blocks_still_hand():
    det = _detector(min_motion=0.0015)
    assert _run(det, step=0.0) == (None, 0.0)
    assert det.last_motion == 0.0


def test_gate_lets_moving_hand_through():
    det = _detector(min_motion=0.0015)
    label, _ = _run(det, step=0.02)
    assert label in ("X", "Y")
    assert det.last_motion > 0.0015


def test_gate_disabled_by_default():
    assert _run(_detector(min_motion=0.0), step=0.0)[0] in ("X", "Y")


def test_manager_reports_winner_and_motion():
    from src.recognition.detector_manager import DetectorManager

    manager = DetectorManager([_detector(min_motion=0.0)], min_score=0.1,
                              stability_frames=3, cooldown_seconds=0)
    for i in range(WINDOW):
        manager.detect(HolisticFrame(hands=[_hand(i * 0.02)]))
    info = manager.debug_info()  # frame em que a janela enche
    assert info["winner"] == "LSTMGestureDetector"
    assert info["trace"][0]["motion"] > 0


class _Always:
    """Detector que sempre responde 'C' com alta confianca."""

    def __init__(self, label="C"):
        self.label = label

    def detect(self, frame):
        return self.label, 0.95


def test_cooldown_and_hold_are_time_based():
    from src.recognition.detector_manager import DetectorManager

    now = [0.0]
    det = _Always()
    mgr = DetectorManager([det], min_score=0.7, stability_frames=2,
                          cooldown_seconds=1.0, hold_seconds=0.5, clock=lambda: now[0])
    frame = HolisticFrame(hands=[_hand()])

    assert mgr.detect(frame) == (None, 0.0)           # 1o frame: ainda estabilizando
    assert mgr.detect(frame)[0] == "C"                # 2o frame: estavel -> cooldown
    det.label = "D"
    now[0] = 0.9                                      # dentro do cooldown de 1 s
    assert mgr.detect(frame)[0] == "C"                # segura o gesto anterior
    now[0] = 1.1                                      # cooldown acabou
    assert mgr.detect(frame)[0] == "C"                # D ainda estabilizando
    assert mgr.detect(frame)[0] == "D"                # D vence (nao ficou preso em C)


def test_hold_expires_after_hold_seconds():
    from src.recognition.detector_manager import DetectorManager

    now = [0.0]
    det = _Always()
    mgr = DetectorManager([det], min_score=0.7, stability_frames=1,
                          cooldown_seconds=0, hold_seconds=0.5, clock=lambda: now[0])
    frame = HolisticFrame(hands=[_hand()])
    assert mgr.detect(frame)[0] == "C"
    det.detect = lambda f: (None, 0.0)                # perdeu a confianca
    now[0] = 0.3
    assert mgr.detect(frame)[0] == "C"                # ainda dentro do hold
    now[0] = 0.7
    assert mgr.detect(frame) == (None, 0.0)           # hold de 0.5 s expirou
