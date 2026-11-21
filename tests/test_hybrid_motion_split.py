"""
Arbitragem estático x dinâmico no modo híbrido pelo movimento da janela.

Logs de 15-20/11 (Logs - APP - Diagnostico): os dois modelos saturam o softmax
em ~1.0. Com "maior score vence":
- SIM real (motion ~0.01) perdia para A/N do estático em quase todo frame;
- com um F parado (motion 0.0015-0.003) o dinâmico dizia OI 1.000 e, quando o
  estático oscilava, o OI era confirmado no lugar da letra.
"""
from collections import deque

from src.recognition.detector_manager import DetectorManager

WINDOW = 15
FRAME = [["mao"]]
SPLIT = 0.008


class FakeSequence:
    def __init__(self, label, score, motion):
        self.window_size = WINDOW
        self.buffer = deque(maxlen=WINDOW)
        self.label, self.score, self.motion = label, score, motion
        self.last_motion = None

    def detect(self, frame):
        self.buffer.append(1)
        self.last_motion = None
        if len(self.buffer) < self.window_size:
            return None, 0.0
        self.last_motion = self.motion
        return self.label, self.score


class FakeStatic:
    def __init__(self, label, score):
        self.label, self.score = label, score

    def detect(self, hand):
        return self.label, self.score


def _run(seq, static, frames=25, **kw):
    now = [0.0]
    params = dict(min_score=0.7, stability_frames=3, cooldown_seconds=1.0,
                  hold_seconds=0.5, hybrid_motion_split=SPLIT)
    params.update(kw)
    mgr = DetectorManager([seq, static], clock=lambda: now[0], **params)
    labels = []
    for i in range(frames):
        now[0] = i * 0.3
        labels.append(mgr.detect(FRAME)[0])
    return labels, mgr


def test_moving_dynamic_beats_saturated_static():
    # SIM real: motion 0.0098-0.015, estático dizendo A com 1.0
    labels, _ = _run(FakeSequence("SIM", 0.999, 0.012), FakeStatic("A", 1.0))
    assert "SIM" in labels


def test_old_behavior_loses_dynamic_on_tie():
    # sem o split, o empate fica com o estático e o SIM nunca aparece
    labels, _ = _run(FakeSequence("SIM", 0.999, 0.012), FakeStatic("A", 1.0),
                     hybrid_motion_split=None)
    assert "SIM" not in labels


def test_still_pose_keeps_static_letter():
    # F parado: OI 1.000 com motion 0.002 não pode roubar a letra
    labels, mgr = _run(FakeSequence("OI", 1.0, 0.002), FakeStatic("F", 0.9))
    assert "OI" not in labels
    assert labels[-1] == "F"


def test_still_pose_dynamic_does_not_fire_when_static_drops():
    # estático sem label (oscilou) e dinâmico de mão parada: nada confirmado
    labels, mgr = _run(FakeSequence("OI", 1.0, 0.005), FakeStatic(None, 0.0))
    assert "OI" not in labels
    gated = [e for e in mgr.debug_info()["trace"] if e.get("gated")]
    assert gated and gated[0]["detector"] == "FakeSequence"


def test_split_ignored_without_static_detectors():
    # modo só-dinâmico: quem filtra é o min_motion do próprio detector
    seq = FakeSequence("OI", 0.99, 0.002)
    now = [0.0]
    mgr = DetectorManager([seq], min_score=0.7, stability_frames=3,
                          cooldown_seconds=0, hybrid_motion_split=SPLIT,
                          clock=lambda: now[0])
    labels = []
    for i in range(WINDOW + 3):
        now[0] = i * 0.3
        labels.append(mgr.detect(FRAME)[0])
    assert "OI" in labels
