import os

import numpy as np
from tensorflow.keras.models import load_model

from src.Bahdanau import BahdanauAttention
from src.losses import BinaryFocalLoss

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_THRESH_PATH = os.path.join(_ROOT, "data", "processed", "inference_threshold.json")

_CUSTOM_OBJECTS = {
    "BahdanauAttention": BahdanauAttention,
    "BinaryFocalLoss": BinaryFocalLoss,
}


def load_trained_model(path=None):
    if path is None:
        for name in ("final_model.h5", "best_model.h5"):
            p = os.path.join(_ROOT, "models", name)
            if os.path.exists(p):
                path = p
                break
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "Aucun modèle trouvé dans models/ (final_model.h5 ou best_model.h5)."
        )
    return load_model(path, custom_objects=_CUSTOM_OBJECTS, compile=False)


def _decision_threshold():
    try:
        import json

        with open(_THRESH_PATH, encoding="utf-8") as f:
            return float(json.load(f).get("decision_threshold", 0.5))
    except Exception:
        return 0.5


def predict(X):
    model = load_trained_model()
    y_score = model.predict(X, verbose=0).ravel()
    y_pred = (y_score >= _decision_threshold()).astype(int)
    return y_pred, y_score
