from collections import deque
import logging
import time

from src.diagnostics import finger_crossing

logger = logging.getLogger(__name__)


class DetectorManager:
    """
    Orquestra múltiplos detectores de gestos.

    Responsabilidades
    -----------------
    - Executar detectores registrados na ordem correta
    - Detectores de sequência (SequenceGestureDetector) recebem
      a lista completa de mãos (hands), detectores estáticos
      recebem uma mão por vez.
    - Aplicar threshold de confiança
    - Estabilizar detecções no tempo
    """

    # Valores padrao no nivel da classe: instancias criadas sem __init__ (testes)
    # continuam funcionando.
    cooldown_until = 0.0
    last_cross = []
    last_hit = 0.0
    cooldown_seconds = 1.0
    hold_seconds = 0.5
    _clock = staticmethod(time.monotonic)

    def __init__(
        self,
        detectors,
        min_score=0.6,
        stability_frames=3,
        cooldown_seconds=1.0,
        hold_seconds=0.5,
        clock=None,
    ):
        self.detectors = detectors
        self.min_score = min_score

        # estabilização temporal
        self.history = deque(maxlen=stability_frames)

        # cooldown (em SEGUNDOS, nao em frames: o aparelho entrega ~3 frames/s,
        # entao 10 frames travavam o mesmo gesto na tela por varios segundos)
        self.cooldown_seconds = cooldown_seconds
        self.cooldown_until = 0.0

        # memória do último gesto detectado
        self.last_label = None
        self.last_score = 0.0

        # Tempo sem detecção confiável. Passando de `hold_seconds`, o último
        # gesto é esquecido; sem isso a tela ficava presa na última letra (A/C)
        # enquanto o modelo não alcançava o limiar de confiança.
        self.hold_seconds = hold_seconds
        self.last_hit = 0.0

        if clock is not None:
            self._clock = clock

        # Diagnostico do ultimo frame: o que cada detector respondeu e quem
        # venceu. Alimenta o log de inferencia do app.
        self.last_trace = []
        self.last_winner = None
        self.last_cooldown = False
        self.last_cross = []

    def debug_info(self) -> dict:
        """Resumo serializavel da ultima decisao (para log/diagnostico)."""
        return {
            "winner": self.last_winner,
            "cooldown": self.last_cooldown,
            "trace": self.last_trace,
            "cross": self.last_cross,
        }

    def _miss(self):
        """Registra um frame sem detecção e devolve o gesto ainda retido."""
        if self._clock() - self.last_hit > self.hold_seconds:
            self.last_label = None
            self.last_score = 0.0
        return self.last_label, self.last_score

    def _is_sequence_detector(self, det) -> bool:
        """
        Verifica se o detector é do tipo sequência (SequenceGestureDetector).
        Detectores de sequência esperam a lista completa de mãos (mãos × frames),
        enquanto detectores estáticos esperam uma única mão (lista de landmarks).
        """
        return hasattr(det, "buffer") and hasattr(det, "window_size")

    def detect(self, frame):
        """
        Executa detectores e retorna gesto estabilizado.

        Parameters
        ----------
        frame : HolisticFrame | list[list[MockLandmark]]
            Frame holístico (mãos + pose/face opcionais). Por compatibilidade,
            também aceita a lista de mãos crua (tratada como frame só-mãos).

        Returns
        -------
        tuple[str|None, float]
            (label, score) — label pode ser None se nada detectado.
        """
        # Detectores de sequência recebem o frame inteiro (mãos + pose + face);
        # detectores estáticos/regras recebem apenas as mãos.
        hands = getattr(frame, "hands", frame)

        self.last_trace = []
        self.last_winner = None
        # Diagnostico de dedos cruzados (so' mede; nao influencia a decisao).
        self.last_cross = finger_crossing.summarize(hands)
        now = self._clock()
        self.last_cooldown = now < self.cooldown_until

        # -------------------------------------------
        # Cooldown ativo — retorna último gesto estável
        # -------------------------------------------
        if self.last_cooldown:
            # Os detectores de sequencia continuam recebendo os frames: a janela
            # deles (15 frames, ~4,5 s no aparelho) precisa ser continua. Sem
            # isto, no modo hibrido cada letra estavel abria 1 s de buraco na
            # janela e, na volta, o intervalo era lido como "a mao saiu e
            # voltou" (reentry_gap_seconds), zerando o buffer de novo.
            self._feed_sequence_detectors(frame)
            return self.last_label, self.last_score

        if not hands:
            self.history.clear()
            self.last_label = None
            self.last_score = 0.0
            return None, 0.0

        best_label = None
        best_score = 0.0
        best_det = None

        # -------------------------------------------
        # Executa detectores
        # -------------------------------------------
        for det in self.detectors:
            try:
                if self._is_sequence_detector(det) or getattr(det, "expects_holistic_frame", False):
                    # Detectores de sequência e estáticos holísticos processam o
                    # frame inteiro (mãos + pose + face). Detectores que só
                    # entendem mãos ignoram os canais extras.
                    label, score = det.detect(frame)
                else:
                    # Detectores estáticos: roda para cada mão, pega o melhor
                    label, score = None, 0.0
                    for hand in hands:
                        try:
                            lbl, scr = det.detect(hand)
                            if lbl and scr > score:
                                label, score = lbl, scr
                        except Exception as e:
                            logger.debug(
                                "[DetectorManager] Erro em detector estático por mão: %s — %s",
                                type(det).__name__, e
                            )

                entry = {
                    "detector": type(det).__name__,
                    "label": str(label) if label else None,
                    "score": round(float(score), 4),
                }
                motion = getattr(det, "last_motion", None)
                if motion is not None:
                    entry["motion"] = round(float(motion), 5)
                self.last_trace.append(entry)

                if label and score > best_score:
                    best_label = label
                    best_score = score
                    best_det = det
                    self.last_winner = type(det).__name__

            except Exception as e:
                logger.warning(
                    "[DetectorManager] Erro em detector %s: %s",
                    type(det).__name__, e
                )
                continue

        # -------------------------------------------
        # Threshold mínimo
        # -------------------------------------------
        if best_score < self.min_score:
            self.history.clear()
            return self._miss()

        self.last_hit = now

        # -------------------------------------------
        # Estabilização temporal
        # -------------------------------------------
        self.history.append(best_label)

        if len(self.history) < self.history.maxlen:
            return self.last_label, self.last_score

        # verifica se todos frames detectaram o mesmo gesto
        if len(set(self.history)) == 1:

            label = self.history[0]

            # salva como último gesto
            self.last_label = label
            self.last_score = best_score

            # ativa cooldown
            self.cooldown_until = now + self.cooldown_seconds
            self.history.clear()

            # Evita o "ghosting" de gestos dinâmicos: limpa os buffers para
            # não ficarem "presos" prevendo a mesma sequência de frames passados.
            # Só quando quem venceu foi um detector de sequência: uma letra
            # estática confirmada não diz nada sobre a janela do dinâmico, e
            # limpar aqui impedia a janela de encher no modo híbrido (o estático
            # confirma uma letra a cada ~3 frames; a janela precisa de 15).
            if best_det is not None and self._is_sequence_detector(best_det):
                for det in self.detectors:
                    if self._is_sequence_detector(det):
                        det.buffer.clear()

            return label, best_score

        return self.last_label, self.last_score

    def _feed_sequence_detectors(self, frame):
        """Entrega o frame aos detectores de sequência, descartando o resultado."""
        for det in self.detectors:
            if not self._is_sequence_detector(det):
                continue
            try:
                det.detect(frame)
            except Exception as e:
                logger.warning(
                    "[DetectorManager] Erro em detector %s: %s",
                    type(det).__name__, e
                )