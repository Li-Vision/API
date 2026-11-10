"""
Treino do classificador estatico pela FORMA DA MAO (`hand_shape_v1`).

Logica pura (numpy/sklearn), sem Supabase, para poder ser testada sozinha. O
`TrainingService` busca as amostras e chama `train_hand_shape_classifier`.

Duas decisoes que mudam o resultado na pratica:

1. Features: so' a forma da mao (ver `hand_shape_features`), nao o vetor
   holistico inteiro.
2. Medida: o teste e' feito com SESSOES de coleta que o treino nao viu. A
   coleta grava dezenas de quadros quase iguais em sequencia; sorteando
   amostras soltas, o teste recebe "copias" do treino e mostra ~99% mesmo
   quando o modelo erra com outra pessoa.
"""

from datetime import datetime
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_collection import hand_shape_features as hs

# Pausa entre amostras da mesma pessoa e letra que separa duas sessoes.
SESSION_GAP_SECONDS = 30.0


def _to_timestamp(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def session_groups(labels, users, times, gap_seconds: float = SESSION_GAP_SECONDS) -> np.ndarray:
    """
    Numero da sessao de coleta de cada amostra.

    Uma sessao e' uma sequencia da mesma pessoa gravando a mesma letra sem
    pausa maior que `gap_seconds`.
    """
    groups = np.zeros(len(labels), dtype=int)
    last = {}  # (user, label) -> [ultimo instante, sessao]
    next_id = 0
    for i in np.argsort(times, kind="stable"):
        key = (users[i], labels[i])
        state = last.get(key)
        if state is None or times[i] - state[0] > gap_seconds:
            next_id += 1
            state = [times[i], next_id]
            last[key] = state
        state[0] = times[i]
        groups[i] = state[1]
    return groups


def session_holdout(labels, groups, times, test_fraction: float = 0.2):
    """
    Separa treino/teste deixando sessoes INTEIRAS de fora.

    Por letra, as sessoes mais recentes vao para o teste ate' somarem
    `test_fraction` das amostras. Uma sessao grande demais (ou a unica da
    letra) e' cortada no tempo: o trecho final vai para o teste, o que ainda
    evita misturar quadros vizinhos entre treino e teste.

    Devolve (indices de treino, indices de teste). Letras com 1 amostra ficam
    so' no treino.
    """
    labels = np.asarray(labels)
    test_mask = np.zeros(len(labels), dtype=bool)

    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        n = len(idx)
        if n < 2:
            continue
        target = max(1, int(round(n * test_fraction)))

        sessions = {}
        for i in idx:
            sessions.setdefault(groups[i], []).append(i)
        # mais recentes primeiro
        ordered = sorted(sessions.values(), key=lambda s: max(times[i] for i in s), reverse=True)

        taken = 0
        for sess in ordered:
            if taken >= target:
                break
            whole_fits = len(ordered) > 1 and taken + len(sess) <= n * 0.4
            if whole_fits:
                test_mask[sess] = True
                taken += len(sess)
            else:
                by_time = sorted(sess, key=lambda i: times[i])
                cut = by_time[len(by_time) - (target - taken):]
                test_mask[cut] = True
                taken += len(cut)

    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def _balanced_indices(labels, rng) -> np.ndarray:
    """Repete amostras das letras com menos exemplos ate' o tamanho da maior."""
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    target = int(counts.max())
    parts = []
    for label, n in zip(classes, counts):
        idx = np.flatnonzero(labels == label)
        parts.append(idx)
        if n < target:
            parts.append(rng.choice(idx, size=target - n, replace=True))
    return np.concatenate(parts)


def _fit(points, labels, seed: int = 42) -> Pipeline:
    rng = np.random.default_rng(seed)
    idx = _balanced_indices(labels, rng)
    aug_points, aug_labels = hs.augment_points(points[idx], np.asarray(labels)[idx], rng)
    X = hs.features_from_points(aug_points)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            max_iter=300,
            random_state=seed,
        )),
    ])
    clf.fit(X, aug_labels)
    clf.feature_schema_ = hs.SCHEMA_HAND_SHAPE_V1
    return clf


def train_hand_shape_classifier(rows, aspect: Optional[float] = None) -> dict:
    """
    Treina o classificador de letras a partir de amostras `holistic_v1`.

    Parameters
    ----------
    rows : list[dict]
        Amostras com `label` e `features` (vetor holistico gravado na coleta);
        `user_id` e `created_at` sao opcionais e servem para montar as sessoes.

    Returns
    -------
    dict com `model` (Pipeline pronto para joblib), `accuracy` (None se nao
    houve teste), `report` (texto), `total_samples`, `used_samples`,
    `classes` e `evaluated` (bool).
    """
    points, labels, users, times = [], [], [], []
    for row in rows:
        pts = hs.points_from_holistic_vector(row["features"], 0, aspect)
        if pts is None:
            continue  # amostra sem mao: nada a aprender
        points.append(pts)
        labels.append(str(row["label"]))
        users.append(row.get("user_id") or "")
        times.append(_to_timestamp(row.get("created_at")))

    if len(points) < 2:
        raise ValueError("Amostras insuficientes para treinar modelo static (minimo 2).")

    points = np.asarray(points)
    labels = np.asarray(labels)
    times = np.asarray(times, dtype=float)
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError(
            "O treino precisa de pelo menos 2 sinais diferentes; o dataset so' tem "
            f"'{classes[0]}'."
        )

    groups = session_groups(labels, users, times)
    train_idx, test_idx = session_holdout(labels, groups, times)

    counts = {c: int(np.sum(labels == c)) for c in classes}
    n_sessions = {c: len(set(groups[labels == c])) for c in classes}
    resumo = ", ".join(f"{c}: {counts[c]} amostras / {n_sessions[c]} sessao(oes)" for c in classes)

    evaluated = len(test_idx) > 0 and len(set(labels[train_idx])) == len(classes)
    if evaluated:
        probe = _fit(points[train_idx], labels[train_idx])
        y_pred = probe.predict(hs.features_from_points(points[test_idx]))
        acc = float(accuracy_score(labels[test_idx], y_pred))
        report = (
            f"Avaliado em {len(test_idx)} amostras de sessoes de coleta que o treino "
            f"nao viu (treino: {len(train_idx)}). O modelo final usa todas as amostras.\n"
            f"Entrada: forma da mao (hand_shape_v1).\n{resumo}\n\n"
            + classification_report(labels[test_idx], y_pred, zero_division=0)
        )
    else:
        acc = None
        report = (
            "ATENCAO: nao ha' amostras suficientes para separar um teste; nenhuma "
            "acuracia foi medida. Colete mais amostras de cada sinal.\n"
            f"Entrada: forma da mao (hand_shape_v1).\n{resumo}"
        )

    poucas = [c for c in classes if n_sessions[c] < 2 or counts[c] < 60]
    if poucas:
        report += (
            "\n\nPouca variedade em: " + ", ".join(poucas) + ". Grave cada sinal em pelo "
            "menos 2 sessoes (de preferencia com pessoas diferentes) e 60+ amostras."
        )

    model = _fit(points, labels)

    return {
        "model": model,
        "accuracy": acc,
        "report": report,
        "total_samples": len(rows),
        "used_samples": int(len(points)),
        "classes": classes,
        "evaluated": evaluated,
    }
