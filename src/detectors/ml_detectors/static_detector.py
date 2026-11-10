"""
ml_detector.py
---------------------------------
Wrapper de Machine Learning para reconhecimento
de gestos de mão (Libras).

Responsabilidade:
- Carregar modelo treinado (.joblib)
- Converter landmarks em vetor de features
- Retornar letra prevista

NÃO depende de OpenCV.
"""

from dataclasses import replace  # Copia o frame holístico trocando só as mãos
from pathlib import Path  # Para manipulação segura de caminhos de arquivos
import numpy as np       # Para manipulação de arrays numéricos
import joblib            # Para carregar modelos scikit-learn serializados

from src.detectors.base_detector import BaseDetector  # Classe base para todos detectores


def _n_features_in(estimator) -> "int | None":
    """
    Numero de features que o modelo espera na entrada.

    Trata as duas formas que convivem no storage:

    - Modelos NOVOS sao um `Pipeline` (StandardScaler + MLPClassifier). O
      Pipeline expoe `n_features_in_` delegando ao primeiro passo, mas isso
      nao e' garantido para todo transformador; ler o passo inicial
      explicitamente evita depender desse detalhe.
    - Modelos ANTIGOS sao o estimador nu, com o atributo direto.

    Sem este tratamento, `expects_holistic_frame` daria False para todo modelo
    holistico novo e a inferencia cairia no caminho legado de 42 features.
    """
    n_in = getattr(estimator, "n_features_in_", None)
    if n_in is not None:
        return n_in

    steps = getattr(estimator, "steps", None)
    if steps:
        return getattr(steps[0][1], "n_features_in_", None)
    return None


# ======================================================
# DETECTOR DE GESTOS USANDO ML
# ======================================================
class MLGestureDetector:
    """
    Detector de gestos usando modelo treinado.
    
    Funcionalidades:
    - Carrega o modelo ML
    - Converte landmarks do MediaPipe em vetor de features
    - Realiza predição da letra prevista
    """

    def __init__(self, model_path: str):
        """
        Inicializa o detector carregando o modelo .joblib.

        Parameters
        ----------
        model_path : str
            Caminho para o modelo treinado
        """
        model_file = Path(model_path)

        # Valida existência do arquivo de modelo
        if not model_file.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        print(f"[ML] Carregando modelo: {model_path}")
        # Carrega o modelo scikit-learn serializado
        self.model = joblib.load(model_file)

    # ==============================
    # FEATURE ENGINEERING
    # ==============================
    def landmarks_to_features(self, landmarks):
        """
        Converte landmarks do MediaPipe em vetor numérico
        que será usado como entrada pelo modelo ML.

        Parameters
        ----------
        landmarks : list
            Lista de 21 landmarks do MediaPipe

        Returns
        -------
        np.ndarray
            Array numpy de shape (1, N) pronto para predição
        """

        x_calc = []
        y_calc = []
        for landmark in landmarks:
            x_calc.append(landmark.x)
            y_calc.append(landmark.y)

        x_min, x_max = min(x_calc), max(x_calc)
        y_min, y_max = min(y_calc), max(y_calc)
        width, height = x_max - x_min, y_max - y_min

        features = []
        for landmark in landmarks:
            # relative x and y
            rel_x = (landmark.x - x_min) / width if width > 0 else 0
            rel_y = (landmark.y - y_min) / height if height > 0 else 0
            features.extend([rel_x, rel_y])

        # Retorna array 2D (1 amostra, N features)
        return np.array(features).reshape(1, -1)

    # ==============================
    # PREDIÇÃO SIMPLES
    # ==============================
    def predict(self, landmarks):
        """
        Retorna a letra prevista pelo modelo ML.

        Parameters
        ----------
        landmarks : list
            Lista de 21 landmarks do MediaPipe

        Returns
        -------
        str | None
            Letra prevista ou None em caso de erro
        """
        try:
            features = self.landmarks_to_features(landmarks)
            prediction = self.model.predict(features)[0]
            return prediction
        except Exception as e:
            # Em caso de erro na predição, imprime log e retorna None
            print(f"[ML ERROR] {e}")
            return None

    # ==============================
    # PREDIÇÃO COM CONFIANÇA
    # ==============================
    def predict_with_confidence(self, landmarks):
        """
        Retorna tupla (classe, confiança).

        Parameters
        ----------
        landmarks : list
            Lista de 21 landmarks do MediaPipe

        Returns
        -------
        tuple
            (label, confidence) se suportado, caso contrário (label, None)
        """
        # Se o modelo não tem método predict_proba, retorna apenas a predição
        if not hasattr(self.model, "predict_proba"):
            return self.predict(landmarks), None

        features = self.landmarks_to_features(landmarks)

        # Probabilidades de cada classe
        probs = self.model.predict_proba(features)[0]

        # Índice da classe mais provável
        idx = np.argmax(probs)

        label = self.model.classes_[idx]
        confidence = probs[idx]

        return label, float(confidence)


# ======================================================
# ADAPTER PARA ARQUITETURA DO PROJETO
# ======================================================
class MLDetector(BaseDetector):
    """
    Adapter que integra o modelo ML ao sistema de detectors do Li-Vision.

    Funcionalidade:
    - Segue interface de BaseDetector
    - Retorna (label, confidence) compatível com DetectorManager
    """

    def __init__(self, model_path: str, threshold: float = 0.7):
        """
        Parameters
        ----------
        model_path : str
            Caminho para o modelo treinado
        threshold : float
            Confiança mínima para aceitar a predição
        """

        self.model = MLGestureDetector(model_path)
        self.threshold = threshold

    @property
    def uses_hand_shape(self) -> bool:
        """
        True quando o modelo foi treinado com a forma da mão (`hand_shape_v1`).

        O treino grava o formato no próprio modelo (`feature_schema_`), então
        não depende de adivinhar pela dimensão de entrada.
        """
        from src.data_collection import hand_shape_features as hs

        return getattr(self.model.model, "feature_schema_", None) == hs.SCHEMA_HAND_SHAPE_V1

    @property
    def expects_holistic_frame(self) -> bool:
        """
        True quando o detector quer o frame inteiro em vez de uma mão por vez.

        Vale para o modelo de forma da mão (que escolhe entre as mãos do frame)
        e para o holístico antigo, reconhecido pela dimensão de entrada: 42
        features é o vetor legado só-mãos; o holístico soma mãos + pose + face.
        """
        from src.data_collection import holistic_features as hf

        if self.uses_hand_shape:
            return True
        n_in = _n_features_in(self.model.model)
        return n_in == hf.HANDS_FEATURES + hf.POSE_FEATURES + hf.FACE_FEATURES

    def _predict_hand_shape(self, frame):
        """
        Prediz (label, confiança) pela forma de cada mão do frame.

        Cada mão é classificada sozinha e vence a de maior confiança. Corpo,
        rosto e posição na tela não entram: é isso que faz o modelo valer para
        qualquer pessoa, e não só para quem gravou as amostras.
        """
        from src.data_collection import hand_shape_features as hs

        hands = getattr(frame, "hands", None)
        if hands is None:
            hands = [frame]  # recebeu uma mão solta (lista de landmarks)
        aspect = getattr(frame, "aspect", None)

        valid = [hand for hand in hands if hand is not None and len(hand) >= hs.N_POINTS]
        if not valid:
            return None, 0.0

        features = [hs.hand_features(hand, aspect) for hand in valid]
        probs = self.model.model.predict_proba(features)
        row, col = np.unravel_index(int(np.argmax(probs)), probs.shape)
        return self.model.model.classes_[col], float(probs[row, col])

    def _predict_holistic(self, frame, hf):
        """
        Prediz (label, confiança) de um frame holístico.

        Os datasets estáticos são de UMA mão: o bloco da mão 2 fica zerado em
        todo o treino. Se o frame traz duas mãos (a outra mão, ou uma detecção
        falsa do MediaPipe), o modelo recebe um padrão que nunca viu e responde
        com confiança alta, quase sempre "C". Por isso, com 2+ mãos cada uma é
        avaliada sozinha (como mão 1, bloco 2 zerado) e vence a de maior
        confiança. Pose e face são mantidas.
        """
        hands = getattr(frame, "hands", None)
        if hands is not None and len(hands) > 1:
            candidates = [replace(frame, hands=[hand]) for hand in hands]
        else:
            candidates = [frame]

        best_label, best_conf = None, 0.0
        for candidate in candidates:
            features = hf.build_frame_vector(candidate, hf.SCHEMA_HOLISTIC_V1)
            probs = self.model.model.predict_proba([features])[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            if best_label is None or conf > best_conf:
                best_label, best_conf = self.model.model.classes_[idx], conf
        return best_label, best_conf

    def detect(self, landmarks):
        """
        Implementa método detect() da interface BaseDetector.

        Recebe uma mão (modelo legado) ou um HolisticFrame (modelo holístico),
        conforme `expects_holistic_frame`.
        """

        if not landmarks:
            return None, 0.0

        if self.uses_hand_shape:
            label, confidence = self._predict_hand_shape(landmarks)
            return (None, 0.0) if confidence < self.threshold else (label, confidence)

        if self.expects_holistic_frame:
            from src.data_collection import holistic_features as hf

            label, confidence = self._predict_holistic(landmarks, hf)
            return (None, 0.0) if confidence < self.threshold else (label, confidence)

        label, confidence = self.model.predict_with_confidence(landmarks)

        # Caso modelo não retorne probabilidade
        if confidence is None:
            return label, 1.0 if label else 0.0

        # Aplica threshold
        if confidence < self.threshold:
            return None, 0.0

        return label, confidence            