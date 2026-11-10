"""
Features de FORMA DA MAO para os sinais estaticos (alfabeto).

Por que existe
--------------
O modelo estatico era treinado com o vetor holistico inteiro (269 features:
posicao absoluta do pulso + mao + corpo + rosto). Para uma letra, so' a forma
da mao importa; o resto descreve QUEM gravou e ONDE estava na tela. A rede
decorava isso: 99% no teste aleatorio e, com as amostras de outra pessoa,
B = 0% e I = 12% (medido no dataset ALFABETO_HOLISTICO). Usando so' a forma da
mao, o mesmo teste entre pessoas da' 100%.

O vetor `hand_shape_v1` (165 features) de UMA mao:

- 60: os 20 pontos (sem o pulso) em x, y, z, relativos ao pulso e divididos
      pelo tamanho da mao -> nao depende de posicao na tela nem de distancia
      da camera;
- 105: distancias entre 15 pontos-chave (pontas, nos dos dedos, pulso) -> nao
      dependem da inclinacao da mao.

Nada de corpo, rosto ou posicao absoluta entra aqui.

Os datasets `holistic_v1` ja' gravados guardam a mao crua (relativa ao pulso),
entao sao convertidos sem precisar coletar de novo: ver
`points_from_holistic_vector`.
"""

from typing import Optional

import numpy as np

from src.data_collection import holistic_features as hf

SCHEMA_HAND_SHAPE_V1 = "hand_shape_v1"

N_POINTS = 21

# Os landmarks vem normalizados por largura e altura da imagem, que nao sao
# iguais: sem corrigir, a mao fica "achatada" e girar a mao muda a forma.
# 0.75 = 480/640, a imagem em pe' que o app usa na inferencia (ver
# useCameraFormat em app/screens/cam). O app novo envia o valor real no campo
# `aspect`; este e' o padrao para amostras antigas, que nao o guardaram.
DEFAULT_ASPECT = 0.75

# Pulso, pontas dos dedos, base dos dedos, nos do meio e o no' do polegar.
_KEY_POINTS = [0, 4, 8, 12, 16, 20, 5, 9, 13, 17, 6, 10, 14, 18, 3]
_PAIRS = np.triu_indices(len(_KEY_POINTS), 1)

FEATURES = (N_POINTS - 1) * 3 + len(_PAIRS[0])  # 60 + 105 = 165


def _valid_aspect(aspect) -> float:
    try:
        a = float(aspect)
    except (TypeError, ValueError):
        return DEFAULT_ASPECT
    return a if 0.2 <= a <= 5.0 else DEFAULT_ASPECT


def points_from_hand(hand, aspect: Optional[float] = None) -> np.ndarray:
    """21 landmarks (objetos com x, y, z) -> matriz (21, 3) em escala real."""
    a = _valid_aspect(aspect)
    pts = np.array(
        [[lm.x, lm.y, getattr(lm, "z", 0.0) or 0.0] for lm in hand[:N_POINTS]],
        dtype=np.float64,
    )
    pts[:, 0] *= a
    pts[:, 2] *= a  # o z do MediaPipe usa a mesma escala do x
    return pts


def points_from_holistic_vector(vec, hand_index: int = 0,
                                aspect: Optional[float] = None) -> Optional[np.ndarray]:
    """
    Recupera a mao (21, 3) de um vetor `holistic_v1`/`hands_v1` ja' gravado.

    Cada mao ocupa 65 posicoes: pulso absoluto (2) + 21 x (dx, dy, z). Devolve
    None quando o bloco esta' zerado (mao ausente).
    """
    start = hand_index * hf.HAND_FEATURES
    block = np.asarray(vec[start:start + hf.HAND_FEATURES], dtype=np.float64)
    if block.shape[0] < hf.HAND_FEATURES or not np.any(block):
        return None
    a = _valid_aspect(aspect)
    pts = block[2:].reshape(N_POINTS, 3).copy()
    pts[:, 0] *= a
    pts[:, 2] *= a
    return pts


def features_from_points(points) -> np.ndarray:
    """
    (21, 3) -> (165,)  ou  (N, 21, 3) -> (N, 165).

    Os pontos podem estar em qualquer posicao/escala: a funcao centra no pulso
    e divide pelo tamanho da mao (diagonal da caixa em x, y).
    """
    p = np.asarray(points, dtype=np.float64)
    single = p.ndim == 2
    if single:
        p = p[None, :, :]

    p = p - p[:, :1, :]
    size = np.linalg.norm(p[:, :, :2].max(axis=1) - p[:, :, :2].min(axis=1), axis=1)
    p = p / np.maximum(size, 1e-6)[:, None, None]

    key = p[:, _KEY_POINTS]
    dists = np.linalg.norm(key[:, :, None, :] - key[:, None, :, :], axis=3)[:, _PAIRS[0], _PAIRS[1]]
    feats = np.hstack([p[:, 1:].reshape(len(p), -1), dists])
    return feats[0] if single else feats


def hand_features(hand, aspect: Optional[float] = None) -> np.ndarray:
    """Atalho da inferencia: landmarks de uma mao -> vetor de 165 features."""
    return features_from_points(points_from_hand(hand, aspect))


def augment_points(points, labels, rng, rotations: int = 2, max_degrees: float = 20.0,
                   jitter: float = 0.01, mirror: bool = True):
    """
    Multiplica as amostras de treino com variacoes que nao mudam a letra:

    - giro de ate' +-`max_degrees` no plano da imagem (mao inclinada);
    - ruido pequeno em cada ponto (tremor do MediaPipe);
    - espelho horizontal (mao esquerda / mao direita).

    Recebe (N, 21, 3) e devolve (N * (1 + rotations) * (2 se mirror), 21, 3) e
    os rotulos repetidos na mesma ordem.
    """
    p = np.asarray(points, dtype=np.float64)
    y = np.asarray(labels)
    parts, ys = [p], [y]

    size = np.linalg.norm(p[:, :, :2].max(axis=1) - p[:, :, :2].min(axis=1), axis=1)
    for _ in range(rotations):
        ang = np.deg2rad(rng.uniform(-max_degrees, max_degrees, len(p)))
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        q = p.copy()
        q[:, :, 0] = c * p[:, :, 0] - s * p[:, :, 1]
        q[:, :, 1] = s * p[:, :, 0] + c * p[:, :, 1]
        q += rng.normal(0.0, jitter, q.shape) * size[:, None, None]
        parts.append(q)
        ys.append(y)

    out, out_y = np.vstack(parts), np.concatenate(ys)
    if mirror:
        flipped = out.copy()
        flipped[:, :, 0] *= -1.0
        out, out_y = np.vstack([out, flipped]), np.concatenate([out_y, out_y])
    return out, out_y
