"""
Diagnostico de DEDOS CRUZADOS a partir dos landmarks do MediaPipe.

Existe para responder, com dados do app real, qual destas hipoteses vale para
um gesto com dedos cruzados:

  (a) os landmarks NAO cruzam (o MediaPipe "desfaz" o cruzamento) -> o modelo
      recebe a mesma forma do gesto sem cruzar e nao tem como diferenciar;
  (b) os landmarks cruzam, mas o modelo nao aprendeu a diferenca.

Entra no `debug` de cada resposta do servidor (campo `cross`), que o app grava
no log de inferencia como `dbg.cross`. Nao altera nenhuma decisao do modelo.

Por par de dedos devolve [inter, swap, gap, dz]:

- inter: 1 se as polilinhas MCP->PIP->DIP->TIP dos dois dedos se cruzam em x,y;
- swap:  1 se a ordem lateral dos dois dedos na PONTA e' inversa a da BASE
         (o sinal mais direto de que um dedo passou por cima do outro);
- gap:   distancia lateral entre as pontas / tamanho da mao, com sinal na
         convencao da base (negativo = pontas trocadas de lado);
- dz:    z da ponta do 1o dedo menos z da ponta do 2o, / tamanho da mao
         (qual dedo esta' na frente; no MediaPipe, menor z = mais perto).
"""

import itertools
from typing import Optional

import numpy as np

from src.data_collection import hand_shape_features as hs

FINGERS = {
    "pol": (1, 2, 3, 4),
    "ind": (5, 6, 7, 8),
    "med": (9, 10, 11, 12),
    "ane": (13, 14, 15, 16),
    "min": (17, 18, 19, 20),
}
PAIRS = list(itertools.combinations(FINGERS, 2))


def _ccw(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d) -> bool:
    return _ccw(a, b, c) * _ccw(a, b, d) < 0 and _ccw(c, d, a) * _ccw(c, d, b) < 0


def _polylines_cross(p, f1, f2) -> bool:
    for k in range(3):
        for m in range(3):
            if _segments_cross(p[f1[k], :2], p[f1[k + 1], :2], p[f2[m], :2], p[f2[m + 1], :2]):
                return True
    return False


def pair_metrics(p: np.ndarray) -> dict:
    """p: (21, 3) em escala real (ver hand_shape_features.points_from_hand)."""
    axis = p[9, :2] - p[0, :2]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return {}
    axis = axis / norm
    perp = np.array([-axis[1], axis[0]])  # lateral da mao
    size = float(np.linalg.norm(p[:, :2].max(axis=0) - p[:, :2].min(axis=0))) or 1e-6

    out = {}
    for n1, n2 in PAIRS:
        f1, f2 = FINGERS[n1], FINGERS[n2]
        base = float((p[f1[0], :2] - p[f2[0], :2]) @ perp)
        tip = float((p[f1[3], :2] - p[f2[3], :2]) @ perp)
        sign = 1.0 if base >= 0 else -1.0
        out[f"{n1}-{n2}"] = [
            int(_polylines_cross(p, f1, f2)),
            int(base * tip < 0),
            round(sign * tip / size, 3),
            round(float(p[f1[3], 2] - p[f2[3], 2]) / size, 3),
        ]
    return out


def summarize(hands, aspect: Optional[float] = None, with_landmarks: bool = True) -> list:
    """
    Uma entrada por mao detectada: {"m": metricas por par, "lm": 21x3 arredondado}.
    Nunca levanta excecao: diagnostico nao pode derrubar a inferencia.
    """
    result = []
    try:
        for hand in hands or []:
            if not hand or len(hand) < hs.N_POINTS:
                continue
            p = hs.points_from_hand(hand, aspect)
            entry = {"m": pair_metrics(p)}
            if with_landmarks:
                entry["lm"] = [[round(float(v), 3) for v in row] for row in p]
            result.append(entry)
    except Exception:
        return result
    return result
