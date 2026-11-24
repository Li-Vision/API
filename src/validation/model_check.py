"""
"O modelo ativo aceita esta amostra?" — a terceira checagem da validacao.

Usa os modelos ja' carregados pelo `ModelCache` (os ativos no servidor). Para
cada amostra procura um modelo que (1) aceite a dimensao dela e (2) conheca o
sinal declarado; sinal novo, que nenhum modelo conhece, nao tem com o que
comparar e a checagem e' pulada.

Devolve {"model", "predicted", "confidence", "label_prob", "knows_label"}.
"""

import logging
import threading
from typing import Callable, Iterable, Optional

import numpy as np

from src.data_collection import hand_shape_features as hs
from src.validation import sample_validator as sv

logger = logging.getLogger(__name__)

_torch_lock = threading.Lock()
_torch_models: dict = {}  # nome -> (id do checkpoint, modulo pronto)


def _static_input(model, vec: np.ndarray):
    """Vetor de entrada do modelo estatico para a amostra, ou None se incompativel."""
    if getattr(model, "feature_schema_", None) == hs.SCHEMA_HAND_SHAPE_V1:
        if vec.size != sv.HOLISTIC_FRAME:
            return None
        pts = hs.points_from_holistic_vector(vec, 0)
        return None if pts is None else hs.features_from_points(pts)

    n_in = getattr(model, "n_features_in_", None)
    if n_in is None and getattr(model, "steps", None):
        n_in = getattr(model.steps[0][1], "n_features_in_", None)
    return vec if n_in == vec.size else None


def _check_static(models: dict, vec: np.ndarray, label: str, prefer: Iterable[str]) -> Optional[dict]:
    for name in _ordered(models, prefer):
        model = models[name]
        classes = [str(c) for c in getattr(model, "classes_", [])]
        if label not in classes or not hasattr(model, "predict_proba"):
            continue
        x = _static_input(model, vec)
        if x is None:
            continue
        probs = model.predict_proba([x])[0]
        return _result(name, classes, probs, label)
    return None


def _torch_module(name: str, checkpoint: dict):
    from src.detectors.ml_detectors.lstm_model import GestureLSTM

    with _torch_lock:
        cached = _torch_models.get(name)
        if cached and cached[0] == id(checkpoint):
            return cached[1]
        module = GestureLSTM(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint.get("hidden_size", 128),
            num_layers=checkpoint.get("num_layers", 2),
            num_classes=len(checkpoint["classes"]),
            dropout=0.0,
            bidirectional=checkpoint.get("bidirectional", True),
            pooling=checkpoint.get("pooling", "last"),
        )
        module.load_state_dict(checkpoint["model_state_dict"])
        module.eval()
        _torch_models[name] = (id(checkpoint), module)
        return module


def _check_dynamic(models: dict, formats: Callable[[str], str], vec: np.ndarray,
                   label: str, prefer: Iterable[str]) -> Optional[dict]:
    for name in _ordered(models, prefer):
        model = models[name]
        if formats(name) == "pt":
            classes = [str(c) for c in model.get("classes", [])]
            size = model.get("input_size")
            if label not in classes or not size or vec.size % size:
                continue
            window = model.get("window_size") or vec.size // size
            if vec.size != window * size:
                continue
            import torch

            module = _torch_module(name, model)
            with torch.no_grad():
                x = torch.from_numpy(vec.astype(np.float32).reshape(1, window, size))
                probs = torch.softmax(module(x), dim=1)[0].numpy()
            return _result(name, classes, probs, label)

        classes = [str(c) for c in getattr(model, "classes_", [])]
        if label in classes and getattr(model, "n_features_in_", None) == vec.size:
            return _result(name, classes, model.predict_proba([vec])[0], label)
    return None


def _ordered(models: dict, prefer: Iterable[str]):
    prefer = [p.upper() for p in prefer]
    return sorted(models, key=lambda n: (n.upper() not in prefer, n))


def _result(name: str, classes, probs, label: str) -> dict:
    idx = int(np.argmax(probs))
    return {
        "model": name,
        "predicted": classes[idx],
        "confidence": round(float(probs[idx]), 4),
        "label_prob": round(float(probs[classes.index(label)]), 4),
        "knows_label": True,
    }


def build_model_check(kind: str, prefer: Iterable[str] = ()) -> Optional[sv.ModelCheck]:
    """
    Funcao `(features, label) -> dict | None` para o tipo de amostra, usando os
    modelos ativos. `prefer`: nomes de modelo treinados com o dataset da
    amostra (testados primeiro). None se nao ha' modelo carregado.
    """
    from src.core.model_cache import ModelCache

    prefer = list(prefer)
    if kind == sv.KIND_STATIC:
        models = ModelCache.get_static_models()
        if not models:
            return None
        return lambda features, label: _check_static(
            models, np.asarray(features, dtype=np.float64).ravel(), label, prefer)

    models = ModelCache.get_dynamic_models()
    if not models:
        return None
    return lambda features, label: _check_dynamic(
        models, ModelCache.get_dynamic_format, np.asarray(features, dtype=np.float64).ravel(), label, prefer)
