import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from src.preprocessing import filter_outliers, engineer_features, impute_missing
from src.Bahdanau import BahdanauAttention

# Constants from preprocessing
ENCOUNTER_ID_COL = "encounter_id"
TIME_COL = "recorded_dttm"
FEATURE_COLS = [
    "fc", "tas", "tad", "fr", "spo2", "temperature",
    "map", "shock_index", "pulse_pressure",
    "lactate", "creatinine", "hemoglobin", "wbc",
    "age", "sex_male", "on_ventilator",
    "fc_trend", "spo2_trend", "tas_trend",
    "fc_std_4", "spo2_std_4",
]

def predict_deterioration(df: pd.DataFrame) -> dict:
    """
    Prédit le risque de détérioration pour un patient basé sur ses données CSV.

    Paramètres :
        df : DataFrame avec colonnes :
            - Vitales : fc, tas, tad, fr, spo2, temperature
            - Labs : lactate, creatinine, hemoglobin, wbc
            - Démographie : age, sex_male, on_ventilator
            - Au moins 24 lignes (6h de données à 15min d'intervalle)

    Retourne :
        dict avec probability, risk_level, timesteps_used, features_engineered
    """
    # Vérifications de base
    if len(df) < 24:
        raise ValueError("Au minimum 24 lignes (6h) sont requises pour la prédiction.")

    required_cols = ["fc", "tas", "tad", "fr", "spo2", "temperature", "age", "sex_male", "on_ventilator"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes requises manquantes : {missing}")

    # Charger les artefacts du modèle
    models_dir = Path("models")
    feature_cols_path = models_dir / "feature_cols.json"
    scaler_path = models_dir / "scaler.pkl"
    imputer_path = models_dir / "imputer.pkl"
    model_path = models_dir / "best_model.h5"

    if not all(p.exists() for p in [feature_cols_path, scaler_path, imputer_path, model_path]):
        raise FileNotFoundError("Artefacts du modèle manquants dans models/")

    with open(feature_cols_path, "r") as f:
        feature_cols = json.load(f)

    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)

    model = load_model(
        model_path,
        custom_objects={"BahdanauAttention": BahdanauAttention}
    )

    # Préparation des données
    df = df.copy()
    df[ENCOUNTER_ID_COL] = "predict_patient"  # ID dummy
    df[TIME_COL] = pd.date_range(start="2023-01-01", periods=len(df), freq="15min")  # Temps dummy

    # Étape 1 : Filtrage des valeurs aberrantes
    df = filter_outliers(df)

    # Étape 2 : Feature engineering
    df = engineer_features(df)

    # Étape 3 : Imputation (utiliser l'imputer chargé)
    available_features = [c for c in feature_cols if c in df.columns]
    df[available_features] = imputer.transform(df[available_features])

    # Étape 4 : Normalisation (utiliser le scaler chargé)
    vals = df[available_features].values.astype(np.float32)
    vals_scaled = scaler.transform(vals.reshape(-1, len(available_features))).reshape(vals.shape)

    # Étape 5 : Construire la séquence (derniers 24 timesteps)
    window_size = 24
    if len(vals_scaled) < window_size:
        # Si moins de 24, prendre tout et padder avec zéros au début
        pad_size = window_size - len(vals_scaled)
        pad = np.zeros((pad_size, len(available_features)))
        sequence = np.vstack([pad, vals_scaled[-window_size:]])
    else:
        sequence = vals_scaled[-window_size:]

    # Reshape pour le modèle (1, 24, n_features)
    X = sequence[np.newaxis, :, :]

    # Prédiction
    prob = model.predict(X, verbose=0).ravel()[0]

    # Niveau de risque
    if prob < 0.5:
        risk_level = "LOW"
    elif prob < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "probability": float(prob),
        "risk_level": risk_level,
        "timesteps_used": len(df),
        "features_engineered": len(available_features),
    }
