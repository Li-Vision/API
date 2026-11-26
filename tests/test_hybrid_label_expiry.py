"""
Validade do gesto exibido.

No log de 26/11 (inference-log-ultima.jsonl), "TUDO BEM" ficou cerca de 5,5 s
na tela enquanto o estatico respondia S, B, R, O, Q e N com confianca alta:
qualquer leitura acima do limiar renovava o prazo do gesto antigo. Agora so'
uma nova leitura do PROPRIO gesto renova.
"""
from collections import deque

from src.recognition.detector_manager import DetectorManager

FRAME = [["mao"]]
STEP = 0.33  # intervalo entre frames no aparelho


class Fake:
    label = None

    def detect(self, hand):
        return self.label, 1.0 if self.label else 0.0


class FakeSequence:
    """Responde `label` em todo frame (janela de 1, para o teste ser curto)."""

    window_size = 1
    label = None
    last_motion = None

    def __init__(self):
        self.buffer = deque(maxlen=1)

    def detect(self, frame):
        return self.label, 1.0 if self.label else 0.0


def _setup(det, **kw):
    now = [0.0]
    params = dict(min_score=0.7, stability_frames=3, cooldown_seconds=1.0,
                  hold_seconds=0.5)
    params.update(kw)
    mgr = DetectorManager([det], clock=lambda: now[0], **params)

    def run(labels):
        out = []
        for label in labels:
            now[0] += STEP
            det.label = label
            out.append(mgr.detect(FRAME)[0])
        return out

    return run


def test_other_confident_labels_do_not_keep_old_label():
    run = _setup(Fake())
    assert run(["A"] * 3)[-1] == "A"

    out = run(["S", "B", "R"] * 5)

    assert out[-1] is None


def test_held_letter_stays_on_screen():
    run = _setup(Fake())
    run(["A"] * 3)

    assert set(run(["A"] * 20)) == {"A"}


def test_single_flicker_after_cooldown_keeps_letter():
    run = _setup(Fake())
    run(["A"] * 3)
    run(["A"] * 4)  # passa o cooldown

    assert set(run(["S", "A", "A", "A"])) == {"A"}


def test_new_stable_letter_replaces_old_one():
    run = _setup(Fake())
    run(["A"] * 3)

    assert run(["B"] * 8)[-1] == "B"


def test_dynamic_gesture_has_minimum_display_time():
    # a janela do dinamico e' apagada ao confirmar, entao o gesto nao e' relido:
    # sem tempo minimo ele sumiria logo apos o cooldown de 1 s
    run = _setup(FakeSequence(), gesture_hold_seconds=2.0)
    assert run(["OI"] * 3)[-1] == "OI"

    out = run([None] * 9)

    assert out[:6] == ["OI"] * 6   # ate' ~1,98 s depois de confirmar
    assert out[-1] is None         # ~2,97 s: ja' expirou


def test_static_letter_gets_no_minimum_display_time():
    run = _setup(Fake(), gesture_hold_seconds=2.0)
    run(["A"] * 3)

    assert run([None] * 5)[-1] is None  # ~1,65 s: passou cooldown (1 s) + hold
