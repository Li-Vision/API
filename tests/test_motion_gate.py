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
