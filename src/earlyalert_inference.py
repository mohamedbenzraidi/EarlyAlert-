"""
Inférence alignée sur le pipeline d'entraînement : 24 timesteps × 13 features
standardisées (StandardScaler du preprocessing), modèle Keras + BahdanauAttention.
"""
from __future__ import annotations

import json
import os

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_PROC = os.path.join(_ROOT, "data", "processed")
_MODELS = os.path.join(_ROOT, "models")

MODEL_FEATURE_ORDER = [
    "HR",
    "SBP",
    "DBP",
    "SpO2",
    "Temp",
    "Resp",
    "Creatinine",
    "Lactate",
    "WBC",
    "shock_index",
    "HR_trend",
    "SpO2_trend",
    "HR_var",
]


def load_scaler():
    try:
        import joblib

        path = os.path.join(_DATA_PROC, "feature_scaler.joblib")
        if os.path.exists(path):
            return joblib.load(path)
    except Exception:
        pass
    return None


def load_training_metrics():
    path = os.path.join(_DATA_PROC, "training_metrics.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_decision_threshold(default=0.5):
    path = os.path.join(_DATA_PROC, "inference_threshold.json")
    if not os.path.exists(path):
        return float(default)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("decision_threshold", default))
    except Exception:
        return float(default)


def load_keras_model():
    import tensorflow as tf

    from src.Bahdanau import BahdanauAttention
    from src.losses import BinaryFocalLoss

    custom_objects = {
        "BahdanauAttention": BahdanauAttention,
        "BinaryFocalLoss": BinaryFocalLoss,
    }
    for name in ("best_model.h5", "final_model.h5"):
        path = os.path.join(_MODELS, name)
        if os.path.exists(path):
            return tf.keras.models.load_model(
                path,
                custom_objects=custom_objects,
                compile=False,
            )
    return None


def demo_ui_sequence_to_raw_model_features(seq_10: np.ndarray) -> np.ndarray:
    """
    Convertit la grille UI démo (24, 10) en 13 colonnes brutes dans l'ordre
    MODEL_FEATURE_ORDER (même logique que preprocessing avant scaler).
    Colonnes démo : FC, TA sys, SpO2, FR, Temp, PVC, Lactate, GCS, Diurèse, FiO2.
    """
    seq = np.asarray(seq_10, dtype=np.float64)
    if seq.shape != (24, 10):
        raise ValueError(f"Séquence démo attendue (24, 10), reçu {seq.shape}.")
    hr = seq[:, 0]
    sbp_raw = seq[:, 1]
    sbp_safe = np.maximum(sbp_raw, 1e-6)
    spo2 = seq[:, 2]
    resp = seq[:, 3]
    temp = seq[:, 4]
    pvc = seq[:, 5]
    lactate = seq[:, 6]
    out = np.zeros((24, 13), dtype=np.float64)
    out[:, 0] = hr
    out[:, 1] = sbp_raw
    out[:, 2] = pvc
    out[:, 3] = spo2
    out[:, 4] = temp
    out[:, 5] = resp
    out[:, 6] = 0.0
    out[:, 7] = lactate
    out[:, 8] = 0.0
    out[:, 9] = hr / sbp_safe
    out[:, 10] = np.concatenate([[0.0], np.diff(hr)])
    out[:, 11] = np.concatenate([[0.0], np.diff(spo2)])
    for t in range(24):
        start = max(0, t - 3)
        out[t, 12] = float(np.std(hr[start : t + 1]))
    return out


def prepare_tensor_for_model(sequence: np.ndarray, scaler) -> np.ndarray:
    """
    Retourne un tenseur (1, 24, 13) float32 prêt pour model.predict.
    - (24, 13) : valeurs brutes alignées sur MODEL_FEATURE_ORDER
    - (24, 10) : conversion démo → 13 puis scaler si disponible
    """
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[0] != 24:
        raise ValueError("Une fenêtre de 24 timesteps est requise.")
    n_feat = sequence.shape[1]
    if n_feat == 10:
        raw = demo_ui_sequence_to_raw_model_features(sequence)
    elif n_feat == 13:
        raw = sequence.copy()
    else:
        raise ValueError(
            f"Nombre de features inattendu : {n_feat} (attendu 10 pour l'UI démo ou 13 pour le pipeline)."
        )
    if scaler is not None:
        scaled = scaler.transform(raw).astype(np.float32)
    else:
        scaled = raw.astype(np.float32)
    return scaled[np.newaxis, ...]


def predict_risk_score(model, sequence: np.ndarray, scaler) -> float:
    x = prepare_tensor_for_model(sequence, scaler)
    return float(model.predict(x, verbose=0)[0, 0])
