"""
Modo híbrido: a janela do detector dinâmico precisa encher mesmo com o
detector estático confirmando letras o tempo todo.

No log de 15/11 (inference-log.jsonl), em 645 frames de modo híbrido a janela
de 15 frames só ficou cheia 6 vezes: cada letra confirmada limpava o buffer do
dinâmico (64 vezes depois de apenas 3 frames) e os frames em cooldown nem
chegavam a ele.
"""
from collections import deque

from src.recognition.detector_manager import DetectorManager

WINDOW = 15
FRAME = [["mao"]]  # uma mão; os detectores falsos ignoram o conteúdo


class FakeSequence:
    """Detector de sequência: só responde com a janela cheia."""

    def __init__(self, label="OI", score=0.99):
        self.window_size = WINDOW
        self.buffer = deque(maxlen=WINDOW)
        self.label, self.score = label, score

    def detect(self, frame):
        self.buffer.append(1)
        if len(self.buffer) < self.window_size:
            return None, 0.0
        return self.label, self.score


class FakeStatic:
    """Detector estático: sempre enxerga a mesma letra."""

    def __init__(self, label="A", score=0.9):
        self.label, self.score = label, score

    def detect(self, hand):
        return self.label, self.score


def _manager(seq, static, now, **kw):
    params = dict(min_score=0.7, stability_frames=3, cooldown_seconds=1.0, hold_seconds=0.5)
    params.update(kw)
    return DetectorManager([seq, static], clock=lambda: now[0], **params)


def test_static_commit_keeps_dynamic_buffer():
    now = [0.0]
    seq = FakeSequence()
    mgr = _manager(seq, FakeStatic(), now, cooldown_seconds=0)

    for i in range(3):
        now[0] = i * 0.3
        label, _ = mgr.detect(FRAME)

    assert label == "A"            # letra confirmada no 3o frame
    assert len(seq.buffer) == 3    # e a janela do dinâmico continua acumulando


def test_cooldown_frames_reach_dynamic_detector():
    now = [0.0]
    seq = FakeSequence()
    mgr = _manager(seq, FakeStatic(), now, stability_frames=1)

    assert mgr.detect(FRAME)[0] == "A"   # confirma e entra em cooldown de 1 s
    now[0] = 0.3
    assert mgr.detect(FRAME)[0] == "A"   # em cooldown: devolve a letra retida
    assert mgr.last_cooldown is True
    assert len(seq.buffer) == 2          # mas o frame entrou na janela


def test_dynamic_commit_still_clears_buffer():
    now = [0.0]
    seq = FakeSequence()
    mgr = _manager(seq, FakeStatic(score=0.0), now, cooldown_seconds=0)

    labels = []
    for i in range(WINDOW + 2):
        now[0] = i * 0.3
        labels.append(mgr.detect(FRAME)[0])

    assert labels[-1] == "OI"
    assert len(seq.buffer) == 0    # anti-ghosting preservado


def test_dynamic_gesture_is_recognized_in_hybrid():
    now = [0.0]
    seq = FakeSequence(score=0.99)
    mgr = _manager(seq, FakeStatic(score=0.9), now)

    labels = []
    for i in range(25):            # ~7,5 s a 3,3 frames/s
        now[0] = i * 0.3
        labels.append(mgr.detect(FRAME)[0])

    assert "OI" in labels
