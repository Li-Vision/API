"""
Validacao de amostras de gesto: integridade + semelhanca com o dataset + modelo.

Logica pura (numpy), sem Supabase, para ser testada sozinha e reaproveitada
pela fila de coleta (`services/collection_queue.py`) e pelo script de
auditoria (`scripts/audit_dataset.py`).

Tres perguntas, em ordem de custo:

1. INTEGRIDADE — a amostra e' fisicamente possivel? NaN/Inf, tamanho que nao
   bate com o schema, mao ausente ou colapsada num ponto, landmarks fora da
   imagem, sequencia congelada (frames repetidos), salto de rastreamento.
   Falhou aqui => REJEITADA: nao ha' o que aprender com ela.

2. SEMELHANCA — a amostra parece com as outras do MESMO sinal no dataset?
   Cada sinal tem um "centro" (mediana por dimensao) e uma dispersao tipica
   (mediana e MAD das distancias das amostras ao centro). A distancia da nova
   amostra vira um z robusto. Mediana/MAD em vez de media/desvio porque o
   proprio dataset pode ja' conter lixo; com estatistica robusta ele nao
   arrasta o centro.
   Muito longe do proprio sinal, ou mais perto de OUTRO sinal => SUSPEITA.

3. MODELO — o modelo ativo reconhece a amostra como outro sinal, com
   confianca alta? Sozinho isso NAO barra a amostra: as amostras em que o
   modelo erra sao justamente as mais valiosas para o proximo treino. So'
   vira SUSPEITA quando a semelhanca tambem ja' estava baixa.

Suspeita nao e' descarte: vai para quarentena e um admin decide.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from src.data_collection import hand_shape_features as hs
from src.data_collection import holistic_features as hf

KIND_STATIC = "static"
KIND_DYNAMIC = "dynamic"

STATUS_ACCEPTED = "accepted"
STATUS_SUSPECT = "suspect"
STATUS_REJECTED = "rejected"

# Vetor estatico legado: 21 pontos (x, y) relativos a caixa da mao.
LEGACY_STATIC_SIZE = 42
HOLISTIC_FRAME = hf.frame_size(hf.SCHEMA_HOLISTIC_V1)
# Tamanhos de frame dinamico cujo bloco de maos segue o layout hands_v1
# (pulso absoluto + 21 x (dx, dy, z) por mao): os unicos que sabemos auditar
# alem do basico.
_DYNAMIC_FRAME_SIZES = (HOLISTIC_FRAME, hf.HANDS_FEATURES)


@dataclass
class ValidationConfig:
    # Amostras do mesmo sinal necessarias para comparar. Abaixo disso a
    # dispersao do sinal ainda nao e' conhecida e so' a integridade vale.
    min_reference: int = 8
    # z robusto (distancia ao centro do sinal, em MADs) a partir do qual a
    # amostra e' considerada fora do padrao. 4 MADs ~ 2.7 desvios de uma
    # normal: raro para uma amostra boa, comum para gesto errado.
    suspect_z: float = 4.0
    # z "moderado": sozinho nao barra, mas confirma uma suspeita do modelo
    # ou de rotulo trocado.
    soft_z: float = 2.5
    # Rotulo trocado: a amostra esta' dentro da dispersao tipica de outro
    # sinal (z <= isto) e mais perto dele do que do proprio.
    other_label_z: float = 1.0
    # Modelo: confianca minima no OUTRO sinal e probabilidade maxima do
    # sinal declarado para contar como discordancia.
    model_confidence: float = 0.90
    model_label_prob: float = 0.05
    # Integridade
    min_hand_size: float = 0.02         # diagonal da caixa da mao (unid. de imagem)
    min_distinct_points: int = 18       # de 21; menos que isso = pontos colapsados
    max_abs_z: float = 1.5              # z do MediaPipe fica em ~+-0.3
    max_out_of_frame: float = 0.30      # fracao de pontos fora de [0, 1]
    max_frames_without_hand: float = 0.20
    max_repeated_frames: float = 0.50   # frames identicos ao anterior
    max_wrist_jump: float = 0.40        # salto do pulso entre frames (unid. de imagem)


@dataclass
class Issue:
    code: str
    message: str
    severity: str  # "error" (rejeita) | "suspect" (quarentena) | "info"

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass
class Verdict:
    status: str
    issues: List[Issue] = field(default_factory=list)
    scores: Dict[str, object] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reasons": [i.to_dict() for i in self.issues],
            "scores": self.scores,
        }


# ---------------------------------------------------------------------------
# Formato
# ---------------------------------------------------------------------------

def dynamic_frame_size(total: int, window: Optional[int] = None) -> Optional[int]:
    """Features por frame de uma sequencia achatada, ou None se nao reconhecido."""
    for size in (HOLISTIC_FRAME, 130, 126, 65, 63, 84, 42):
        if total % size == 0 and total // size >= 2:
            if window is None or total // size == window:
                return size
    return None


def _hand_points_from_block(block: np.ndarray) -> Optional[np.ndarray]:
    """Bloco de 65 (pulso absoluto + 21 x rel) -> (21, 3) ABSOLUTO, ou None se zerado."""
    if block.shape[0] < hf.HAND_FEATURES or not np.any(block):
        return None
    rel = block[2:].reshape(hs.N_POINTS, 3).copy()
    rel[:, 0] += block[0]
    rel[:, 1] += block[1]
    return rel


def _legacy_points(vec: np.ndarray) -> np.ndarray:
    """Vetor estatico de 42 (x, y relativos a caixa) -> (21, 3) com z = 0."""
    pts = np.zeros((hs.N_POINTS, 3))
    pts[:, :2] = vec.reshape(hs.N_POINTS, 2)
    return pts


# ---------------------------------------------------------------------------
# 1. Integridade
# ---------------------------------------------------------------------------

def check_hand_points(points: np.ndarray, cfg: ValidationConfig,
                      absolute: bool = True, where: str = "") -> List[Issue]:
    """Problemas de uma mao (21, 3). `absolute`=False desliga a checagem de borda."""
    issues: List[Issue] = []
    prefix = f"{where}: " if where else ""

    xy = points[:, :2]
    diag = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
    if diag < cfg.min_hand_size:
        issues.append(Issue(
            "hand_collapsed",
            f"{prefix}mão colapsada (tamanho {diag:.3f}); os landmarks não formam uma mão.",
            "error",
        ))
        return issues

    distinct = len(np.unique(np.round(xy / max(diag, 1e-9), 3), axis=0))
    if distinct < cfg.min_distinct_points:
        issues.append(Issue(
            "points_merged",
            f"{prefix}só {distinct} de 21 pontos distintos; landmarks sobrepostos.",
            "error",
        ))

    if np.abs(points[:, 2]).max() > cfg.max_abs_z:
        issues.append(Issue(
            "depth_out_of_range",
            f"{prefix}profundidade (z) fora da faixa do MediaPipe.",
            "error",
        ))

    if absolute:
        outside = float(np.mean((xy < -0.02) | (xy > 1.02)))
        if outside > cfg.max_out_of_frame:
            issues.append(Issue(
                "hand_out_of_frame",
                f"{prefix}{outside:.0%} dos pontos fora da imagem; mão cortada pela borda.",
                "suspect",
            ))
    return issues


def check_hand_landmarks(hand, cfg: Optional[ValidationConfig] = None) -> List[Issue]:
    """Atalho para landmarks crus (objetos com x, y, z) recebidos do app."""
    cfg = cfg or ValidationConfig()
    if hand is None or len(hand) < hs.N_POINTS:
        return [Issue("hand_missing", "Nenhuma mão completa (21 pontos) na amostra.", "error")]
    pts = np.array([[lm.x, lm.y, getattr(lm, "z", 0.0) or 0.0] for lm in hand[:hs.N_POINTS]],
                   dtype=np.float64)
    if not np.all(np.isfinite(pts)):
        return [Issue("non_finite", "Landmarks com NaN/Inf.", "error")]
    return check_hand_points(pts, cfg)


def check_integrity(features, kind: str, cfg: Optional[ValidationConfig] = None,
                    window: Optional[int] = None) -> List[Issue]:
    """Problemas de integridade de um vetor ja' montado (como gravado em `samples`)."""
    cfg = cfg or ValidationConfig()
    try:
        vec = np.asarray(features, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return [Issue("not_numeric", "Features não numéricas.", "error")]

    if vec.size == 0:
        return [Issue("empty", "Amostra vazia.", "error")]
    if not np.all(np.isfinite(vec)):
        return [Issue("non_finite", f"{int(np.sum(~np.isfinite(vec)))} valores NaN/Inf.", "error")]
    if not np.any(vec):
        return [Issue("all_zero", "Amostra toda zerada.", "error")]

    if kind == KIND_STATIC:
        return _check_static(vec, cfg)
    return _check_dynamic(vec, cfg, window)


def _check_static(vec: np.ndarray, cfg: ValidationConfig) -> List[Issue]:
    if vec.size == LEGACY_STATIC_SIZE:
        if vec.min() < -1e-6 or vec.max() > 1 + 1e-6:
            return [Issue("out_of_range", "Coordenadas fora de [0, 1] no formato 'só mãos'.", "error")]
        # Relativo a caixa da mao: o tamanho e' sempre ~1.4, a borda nao se aplica.
        return check_hand_points(_legacy_points(vec), cfg, absolute=False)

    if vec.size == HOLISTIC_FRAME:
        pts = _hand_points_from_block(vec[:hf.HAND_FEATURES])
        if pts is None:
            return [Issue("hand_missing", "Amostra sem a mão principal.", "error")]
        return check_hand_points(pts, cfg)

    return [Issue(
        "bad_size",
        f"{vec.size} features; esperado {LEGACY_STATIC_SIZE} (só mãos) ou {HOLISTIC_FRAME} (holístico).",
        "error",
    )]


def _check_dynamic(vec: np.ndarray, cfg: ValidationConfig, window: Optional[int]) -> List[Issue]:
    size = dynamic_frame_size(vec.size, window)
    if size is None:
        return [Issue("bad_size", f"{vec.size} features não formam uma sequência conhecida.", "error")]
    frames = vec.reshape(-1, size)
    n = len(frames)
    issues: List[Issue] = []

    # Frames identicos ao anterior: camera/stream travado reenviando o mesmo frame.
    repeated = float(np.mean(np.all(np.abs(np.diff(frames, axis=0)) < 1e-9, axis=1)))
    if repeated >= 0.999:
        return [Issue("frozen", "Sequência congelada: todos os frames são idênticos.", "error")]
    if repeated > cfg.max_repeated_frames:
        issues.append(Issue(
            "repeated_frames",
            f"{repeated:.0%} dos frames repetem o anterior (stream travado).",
            "error",
        ))

    if size not in _DYNAMIC_FRAME_SIZES:
        return issues  # formato legado: so' o basico

    hands = [_hand_points_from_block(f[:hf.HAND_FEATURES]) for f in frames]
    missing = sum(h is None for h in hands) / n
    if missing > cfg.max_frames_without_hand:
        issues.append(Issue(
            "hand_missing",
            f"Mão principal ausente em {missing:.0%} dos frames.",
            "error",
        ))

    bad_frames = 0
    worst: List[Issue] = []
    for i, pts in enumerate(hands):
        if pts is None:
            continue
        found = [x for x in check_hand_points(pts, cfg, where=f"frame {i + 1}") if x.severity == "error"]
        if found:
            bad_frames += 1
            worst = worst or found
    if bad_frames / n > cfg.max_frames_without_hand:
        issues.append(Issue(
            "bad_frames",
            f"{bad_frames} de {n} frames com mão inválida ({worst[0].message})",
            "error",
        ))

    wrists = np.array([h[0, :2] if h is not None else (np.nan, np.nan) for h in hands])
    jumps = np.linalg.norm(np.diff(wrists, axis=0), axis=1)
    jumps = jumps[np.isfinite(jumps)]
    if jumps.size and jumps.max() > cfg.max_wrist_jump:
        issues.append(Issue(
            "tracking_jump",
            f"Salto de {jumps.max():.2f} da mão entre dois frames (rastreamento trocou de mão?).",
            "suspect",
        ))
    return issues


# ---------------------------------------------------------------------------
# 2. Descritor para comparar amostras
# ---------------------------------------------------------------------------

def _mirror_points(pts: np.ndarray) -> np.ndarray:
    out = pts.copy()
    out[..., 0] *= -1.0
    return out


def descriptors(features, kind: str, window: Optional[int] = None) -> Optional[List[Dict[str, np.ndarray]]]:
    """
    Descritores comparaveis da amostra: [original, espelhado].

    O espelho existe porque canhoto e destro fazem o mesmo sinal com formas
    espelhadas; sem ele quem usa a outra mao viraria "fora do padrao".
    Cada descritor e' um dict componente -> vetor; o z final e' o pior
    componente. None quando o formato nao permite comparar.
    """
    vec = np.asarray(features, dtype=np.float64).ravel()

    if kind == KIND_STATIC:
        if vec.size == LEGACY_STATIC_SIZE:
            pts = vec.reshape(hs.N_POINTS, 2)
            mirrored = pts.copy()
            mirrored[:, 0] = 1.0 - mirrored[:, 0]
            return [{"shape": pts.ravel()}, {"shape": mirrored.ravel()}]
        if vec.size == HOLISTIC_FRAME:
            pts = hs.points_from_holistic_vector(vec, 0)
            if pts is None:
                return None
            return [{"shape": hs.features_from_points(p)} for p in (pts, _mirror_points(pts))]
        return None

    size = dynamic_frame_size(vec.size, window)
    if size not in _DYNAMIC_FRAME_SIZES:
        return None
    frames = vec.reshape(-1, size)
    raw = [hs.points_from_holistic_vector(f, 0) for f in frames]
    valid = [i for i, p in enumerate(raw) if p is not None]
    if len(valid) < max(2, len(raw) // 2):
        return None
    # Frame sem mao (a integridade tolera alguns) herda o frame valido mais
    # proximo; zerado, ele viraria um salto falso na trajetoria.
    fill = [min(valid, key=lambda j: abs(j - i)) for i in range(len(raw))]
    pts = np.asarray([raw[j] for j in fill])    # (T, 21, 3), relativo ao pulso
    wrist = frames[fill, :2] * np.array([hs.DEFAULT_ASPECT, 1.0])
    out = []
    for p, sign in ((pts, 1.0), (_mirror_points(pts), -1.0)):
        shape = hs.features_from_points(p)      # (T, 165)
        # Forma em tres momentos (inicio, meio, fim): resume a evolucao da
        # mao sem depender do frame exato em que cada fase cai.
        thirds = np.array_split(shape, 3)
        shape_desc = np.concatenate([t.mean(axis=0) for t in thirds])
        # Trajetoria do pulso relativa ao 1o frame, em "tamanhos de mao":
        # independe de onde a pessoa estava na tela e da distancia da camera.
        size_hand = np.linalg.norm(p[:, :, :2].max(axis=1) - p[:, :, :2].min(axis=1), axis=1).mean()
        traj = (wrist - wrist[0]) / max(float(size_hand), 1e-6)
        traj[:, 0] *= sign
        out.append({"shape": shape_desc, "trajectory": traj.ravel()})
    return out


# ---------------------------------------------------------------------------
# Referencia do dataset (por sinal)
# ---------------------------------------------------------------------------

class _LabelStats:
    __slots__ = ("items", "_stats")

    def __init__(self):
        self.items: List[Dict[str, np.ndarray]] = []
        self._stats = None

    def add(self, desc: Dict[str, np.ndarray]):
        self.items.append(desc)
        self._stats = None

    def stats(self):
        """componente -> (centro, mediana das distancias, escala)."""
        if self._stats is None:
            out = {}
            for comp in self.items[0]:
                mat = np.stack([d[comp] for d in self.items])
                center = np.median(mat, axis=0)
                dist = np.linalg.norm(mat - center, axis=1)
                med = float(np.median(dist))
                mad = float(np.median(np.abs(dist - med))) * 1.4826
                # Piso na escala: com amostras quase identicas (mesma sessao)
                # o MAD tende a zero e qualquer variacao viraria z enorme.
                scale = max(mad, 0.15 * med, 1e-6)
                out[comp] = (center, med, scale)
            self._stats = out
        return self._stats

    def z(self, desc: Dict[str, np.ndarray]) -> Dict[str, float]:
        return {
            comp: (float(np.linalg.norm(desc[comp] - center)) - med) / scale
            for comp, (center, med, scale) in self.stats().items()
        }


class DatasetReference:
    """
    Distribuicao de cada sinal de um dataset, para medir se uma amostra nova
    "parece" com as que ja' existem.
    """

    def __init__(self, kind: str, window: Optional[int] = None):
        self.kind = kind
        self.window = window
        self.labels: Dict[str, _LabelStats] = {}
        self.counts: Dict[str, int] = {}

    @classmethod
    def from_rows(cls, rows, kind: str, window: Optional[int] = None,
                  cfg: Optional[ValidationConfig] = None) -> "DatasetReference":
        """Monta a referencia ignorando o que ja' esta' corrompido no dataset."""
        cfg = cfg or ValidationConfig()
        ref = cls(kind, window)
        for row in rows:
            label = str(row["label"])
            ref.counts[label] = ref.counts.get(label, 0) + 1
            if any(i.severity == "error" for i in check_integrity(row["features"], kind, cfg, window)):
                continue
            ref.add(label, row["features"], count=False)
        return ref

    def add(self, label: str, features, count: bool = True) -> None:
        if count:
            self.counts[label] = self.counts.get(label, 0) + 1
        descs = descriptors(features, self.kind, self.window)
        if descs is None:
            return
        stats = self.labels.get(label)
        if stats is not None and set(stats.items[0]) != set(descs[0]):
            return  # formato diferente do que ja' existe no sinal
        self.labels.setdefault(label, _LabelStats()).add(descs[0])

    def reference_size(self, label: str) -> int:
        stats = self.labels.get(label)
        return len(stats.items) if stats else 0

    def score(self, features, label: str, cfg: ValidationConfig) -> Optional[dict]:
        """
        z robusto da amostra contra o proprio sinal e contra o sinal mais
        parecido. None se nao ha' referencia suficiente para o sinal.
        """
        own = self.labels.get(label)
        if own is None or len(own.items) < cfg.min_reference:
            return None
        descs = descriptors(features, self.kind, self.window)
        if descs is None or set(descs[0]) != set(own.items[0]):
            return None

        # Variante (original/espelhada) mais proxima do proprio sinal.
        best = min(({"desc": d, "z": own.z(d)} for d in descs), key=lambda v: max(v["z"].values()))
        z_own = max(best["z"].values())

        nearest_label, nearest_z = None, None
        for other, stats in self.labels.items():
            if other == label or len(stats.items) < cfg.min_reference:
                continue
            if set(stats.items[0]) != set(descs[0]):
                continue
            z_other = min(max(stats.z(d).values()) for d in descs)
            if nearest_z is None or z_other < nearest_z:
                nearest_label, nearest_z = other, z_other

        return {
            "z_own": round(z_own, 3),
            "z_components": {k: round(v, 3) for k, v in best["z"].items()},
            "nearest_label": nearest_label,
            "z_nearest": None if nearest_z is None else round(nearest_z, 3),
            "reference_size": len(own.items),
        }


def medoid_index(features_list, kind: str, window: Optional[int] = None) -> Optional[int]:
    """
    Indice da amostra mais "tipica" do grupo: a de menor distancia total as
    demais. Serve de exemplo visual do sinal na revisao da quarentena — uma
    amostra real, nao uma media que pode nao parecer mao nenhuma.
    """
    items = []
    for i, feats in enumerate(features_list):
        if any(x.severity == "error" for x in check_integrity(feats, kind, window=window)):
            continue
        d = descriptors(feats, kind, window)
        if d is not None:
            items.append((i, np.concatenate([d[0][k] for k in sorted(d[0])])))
    if not items:
        return None
    sizes = {len(v) for _, v in items}
    if len(sizes) > 1:
        main = max(sizes, key=lambda s: sum(len(v) == s for _, v in items))
        items = [(i, v) for i, v in items if len(v) == main]
    mat = np.stack([v for _, v in items])
    total = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=2).sum(axis=1)
    return items[int(np.argmin(total))][0]


# ---------------------------------------------------------------------------
# Veredito
# ---------------------------------------------------------------------------

ModelCheck = Callable[[object, str], Optional[dict]]

_COMPONENT_NAMES = {"shape": "forma da mão", "trajectory": "trajetória"}


def validate(features, label: str, kind: str,
             reference: Optional[DatasetReference] = None,
             model_check: Optional[ModelCheck] = None,
             cfg: Optional[ValidationConfig] = None,
             window: Optional[int] = None,
             extra_issues: Optional[List[Issue]] = None) -> Verdict:
    """
    Veredito completo de uma amostra.

    `model_check(features, label)` (opcional) devolve
    {"model", "predicted", "confidence", "label_prob", "knows_label"} ou None.
    `extra_issues`: problemas ja' encontrados nos landmarks crus.
    """
    cfg = cfg or ValidationConfig()
    issues = list(extra_issues or [])
    issues += check_integrity(features, kind, cfg, window)
    scores: Dict[str, object] = {}

    if any(i.severity == "error" for i in issues):
        return Verdict(STATUS_REJECTED, issues, scores)

    sim = reference.score(features, label, cfg) if reference is not None else None
    if sim is None:
        n = reference.reference_size(label) if reference is not None else 0
        issues.append(Issue(
            "few_reference",
            f"Só {n} amostra(s) válidas de '{label}' para comparar (mínimo {cfg.min_reference}); "
            "checada apenas a integridade.",
            "info",
        ))
    else:
        scores["similarity"] = sim
        z_own = sim["z_own"]
        if z_own >= cfg.suspect_z:
            worst = max(sim["z_components"], key=sim["z_components"].get)
            issues.append(Issue(
                "outlier",
                f"Muito diferente das outras amostras de '{label}' "
                f"({_COMPONENT_NAMES.get(worst, worst)}: z={sim['z_components'][worst]:.1f}).",
                "suspect",
            ))
        z_near = sim["z_nearest"]
        if (z_near is not None and z_own >= cfg.soft_z
                and z_near <= cfg.other_label_z and z_near < z_own):
            issues.append(Issue(
                "looks_like_other",
                f"Parece mais com '{sim['nearest_label']}' do que com '{label}' (rótulo trocado?).",
                "suspect",
            ))

    if model_check is not None:
        try:
            mc = model_check(features, label)
        except Exception as exc:  # modelo incompativel nao derruba a coleta
            mc = None
            issues.append(Issue("model_error", f"Checagem do modelo falhou: {exc}", "info"))
        if mc:
            scores["model"] = mc
            disagrees = (
                mc.get("knows_label")
                and mc.get("predicted") != label
                and (mc.get("confidence") or 0.0) >= cfg.model_confidence
                and (mc.get("label_prob") or 0.0) <= cfg.model_label_prob
            )
            if disagrees:
                corroborated = sim is not None and sim["z_own"] >= cfg.soft_z
                issues.append(Issue(
                    "model_disagrees",
                    f"Modelo '{mc.get('model')}' reconhece como '{mc['predicted']}' "
                    f"({mc['confidence']:.0%}); '{label}' tem {mc.get('label_prob', 0):.0%}.",
                    "suspect" if corroborated else "info",
                ))

    status = STATUS_SUSPECT if any(i.severity == "suspect" for i in issues) else STATUS_ACCEPTED
    return Verdict(status, issues, scores)
