"""
Classe NADA: o dinamico a usa para dizer "isto nao e' gesto nenhum" (mao
parada, transicao entre letras, mao entrando no quadro). Ela nunca vira gesto
na tela e nunca tira a vez do estatico.
"""
from collections import deque

from src.recognition.detector_manager import DetectorManager

FRAME = [["mao"]]
SPLIT = 0.008


class FakeSequence:
    window_size = 1

    def __init__(self, label, motion=0.02, score=1.0):
        self.buffer = deque(maxlen=1)
        self.label, self.score, self.motion = label, score, motion
        self.last_motion = None

    def detect(self, frame):
        self.buffer.append(1)
        self.last_motion = self.motion
        return self.label, self.score


class FakeStatic:
    def __init__(self, label, score=0.9):
        self.label, self.score = label, score

    def detect(self, hand):
        return self.label, self.score if self.label else 0.0


def _run(detectors, frames=10, split=SPLIT):
    now = [0.0]
    mgr = DetectorManager(detectors, min_score=0.7, stability_frames=3,
                          cooldown_seconds=1.0, hold_seconds=0.5,
                          hybrid_motion_split=split, clock=lambda: now[0])
    labels = []
    for i in range(frames):
        now[0] = i * 0.33
        labels.append(mgr.detect(FRAME)[0])
    return labels, mgr


def test_nada_with_motion_leaves_the_turn_to_static():
    labels, mgr = _run([FakeSequence("NADA"), FakeStatic("A")])

    assert "NADA" not in labels
    assert labels[-1] == "A"


def test_nada_is_ignored_without_motion_split():
    # sem o split vence o maior score: NADA 1.0 nao pode ganhar de A 0.9
    labels, _ = _run([FakeSequence("NADA"), FakeStatic("A")], split=None)

    assert "NADA" not in labels
    assert labels[-1] == "A"


def test_nada_is_never_shown_in_dynamic_only_mode():
    labels, _ = _run([FakeSequence("NADA")])

    assert set(labels) == {None}


def test_nada_does_not_clear_the_dynamic_window():
    seq = FakeSequence("NADA")
    _run([seq], frames=4)

    assert len(seq.buffer) == 1


def test_nada_is_visible_in_the_diagnostic_trace():
    _, mgr = _run([FakeSequence("NADA"), FakeStatic(None)], frames=1)

    entry = mgr.debug_info()["trace"][0]
    assert entry["label"] == "NADA"
    assert entry["gated"] == "rejection_class"
    assert mgr.debug_info()["winner"] is None


def test_real_gesture_still_wins():
    labels, _ = _run([FakeSequence("OI"), FakeStatic("A")])

    assert "OI" in labels
