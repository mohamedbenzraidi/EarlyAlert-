"""""
Étapes :
    1. Filtrage des valeurs aberrantes (plages physiologiques)
    2. Rééchantillonnage à 15 minutes (grille temporelle régulière)
    3. Forward-fill puis KNNImputer pour les valeurs manquantes
    4. Feature engineering : shock index, MAP, variabilité, tendances
    5. Labellisation : détérioration dans les 6h suivantes ?
    6. Construction des fenêtres glissantes (séquences LSTM)
    7. Normalisation (StandardScaler fit sur train seulement)
    8. Split Train / Validation / Test stratifié
    9. Sauvegarde des arrays numpy

Pourquoi ces choix ?
    - 15 min : granularité standard en réa — assez fin pour détecter des
      changements rapides, assez large pour limiter le bruit
    - 24 timesteps : fenêtre de 6h (24 × 15min) — cohérent avec l'horizon
      de prédiction voulu (6h avant l'événement)
    - Forward-fill avant KNN : les labs (lactate, créat) changent lentement,
      la dernière valeur connue reste la meilleure estimation
    - KNNImputer (k=5) : meilleur que la moyenne car exploite la similarité
      entre patients de même profil
"""

import logging
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import joblib
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocessing")

# ─────────────────────────────────────────────
# Constantes de configuration
# ─────────────────────────────────────────────

ENCOUNTER_ID_COL = "encounter_id"
TIME_COL = "recorded_dttm"

# Fréquence de rééchantillonnage
RESAMPLE_FREQ = "15min"

# Taille de la fenêtre LSTM (24 × 15min = 6h)
WINDOW_SIZE = 24

# Horizon de prédiction : on prédit une détérioration dans les 6h suivantes
PREDICTION_HORIZON_H = 6

# Plages physiologiques valides — valeurs hors plage = bruit / erreur saisie
VITAL_VALID_RANGES = {
    "fc":          (20, 300),     # FC (bpm) : bradycardie sévère à tachycardie extrême
    "tas":         (40, 300),     # TAS (mmHg)
    "tad":         (10, 200),     # TAD (mmHg)
    "fr":          (4, 70),       # FR (/min)
    "spo2":        (50, 100),     # SpO2 (%)
    "temperature": (25, 45),      # Température (°C)
}

LAB_VALID_RANGES = {
    "lactate":     (0.1, 30),     # Lactate (mmol/L)
    "creatinine":  (0.1, 30),     # Créatinine (mg/dL)
    "hemoglobin":  (1, 25),       # Hémoglobine (g/dL)
    "wbc":         (0.1, 100),    # GB (10^9/L)
}

# Features finales utilisées comme input du LSTM
# (dans cet ordre exact → important pour la reproductibilité)
FEATURE_COLS = [
    # Vitales de base
    "fc", "tas", "tad", "fr", "spo2", "temperature",
    # Features dérivées
    "map",          # Pression artérielle moyenne
    "shock_index",  # FC / TAS — marqueur de choc précoce
    "pulse_pressure",  # TAS - TAD
    # Labs (sparse, forward-fillés)
    "lactate", "creatinine", "hemoglobin", "wbc",
    # Démographie & support
    "age", "sex_male", "on_ventilator",
    # Variabilité et tendances (fenêtre glissante sur les 4 dernières mesures)
    "fc_trend", "spo2_trend", "tas_trend",
    "fc_std_4",  "spo2_std_4",
]

# Label : nom de la colonne binaire de détérioration
LABEL_COL = "deterioration_label"

# Split
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


# ─────────────────────────────────────────────
# 1. Filtrage des valeurs aberrantes
# ─────────────────────────────────────────────

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace par NaN les valeurs hors des plages physiologiques.
    On NaN-ise plutôt que de supprimer la ligne entière car les autres
    features du même timestep restent valides.
    """
    log.info("🔍 Filtrage des valeurs aberrantes...")
    total_replaced = 0

    all_ranges = {**VITAL_VALID_RANGES, **LAB_VALID_RANGES}
    for col, (lo, hi) in all_ranges.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n = mask.sum()
        df.loc[mask, col] = np.nan
        total_replaced += n
        if n > 0:
            log.info(f"  {col}: {n:,} valeurs aberrantes → NaN "
                     f"({n/len(df)*100:.1f}%)")

    log.info(f"  Total remplacés : {total_replaced:,} valeurs")
    return df


# ─────────────────────────────────────────────
# 2. Rééchantillonnage à 15 minutes par patient
# ─────────────────────────────────────────────

def resample_patient(group: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """
    Rééchantillonne un seul patient sur une grille temporelle régulière.
    Appliqué via groupby(encounter_id).apply().

    Stratégie :
        - Pour les vitales : moyenne des mesures dans la fenêtre de 15min
        - Pour les labs : forward-fill (la dernière valeur lab reste valide
          jusqu'à la prochaine prise)
    """
    group = group.set_index(TIME_COL).sort_index()
    encounter_id = group[ENCOUNTER_ID_COL].iloc[0]

    # Colonnes vitales : agrégation par moyenne
    vital_cols = [c for c in VITAL_VALID_RANGES if c in group.columns]
    lab_cols   = [c for c in LAB_VALID_RANGES if c in group.columns]
    static_cols = [c for c in ["age", "sex_male", "on_ventilator", "icu_in_dttm", "icu_out_dttm"]
                   if c in group.columns]

    # Rééchantillonnage vitales → moyenne
    df_vitals = group[vital_cols].resample(freq).mean() if vital_cols else pd.DataFrame()

    # Rééchantillonnage labs → on prend la dernière valeur (first() car forward-fill ensuite)
    df_labs = group[lab_cols].resample(freq).first() if lab_cols else pd.DataFrame()

    # Rééchantillonnage features statiques → forward-fill
    df_static = group[static_cols].resample(freq).first() if static_cols else pd.DataFrame()

    # Fusion
    frames = [f for f in [df_vitals, df_labs, df_static] if not f.empty]
    if not frames:
        return pd.DataFrame()
    df_res = pd.concat(frames, axis=1)

    # Forward-fill des labs (valeur précédente valable jusqu'à la prochaine mesure)
    df_res[lab_cols] = df_res[lab_cols].ffill(limit=8)  # max 2h de forward-fill (8 × 15min)

    # Forward-fill des statiques
    df_res[static_cols] = df_res[static_cols].ffill()

    df_res[ENCOUNTER_ID_COL] = encounter_id
    df_res = df_res.reset_index().rename(columns={"index": TIME_COL})
    return df_res


def resample_all(df: pd.DataFrame) -> pd.DataFrame:
    """Applique resample_patient à chaque séjour ICU."""
    log.info(f"⏱️  Rééchantillonnage à {RESAMPLE_FREQ}...")
    groups = [resample_patient(g) for _, g in df.groupby(ENCOUNTER_ID_COL)]
    df_res = pd.concat(groups, ignore_index=True)
    log.info(f"  {len(df_res):,} timesteps après rééchantillonnage")
    return df_res


# ─────────────────────────────────────────────
# 3. Feature Engineering
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée des features dérivées cliniquement pertinentes.

    MAP (Pression Artérielle Moyenne) :
        MAP = TAD + (TAS - TAD) / 3
        → Indicateur de perfusion des organes. MAP < 65 = choc.

    Shock Index :
        SI = FC / TAS
        → SI > 1 = risque de choc hémorragique ou septique élevé
        → Simple mais puissant : prédit la mortalité précocement

    Pulse Pressure (Pression différentielle) :
        PP = TAS - TAD
        → PP étroite (< 25) = signe de bas débit cardiaque

    Tendances (fc_trend, spo2_trend, tas_trend) :
        Différence entre la valeur actuelle et la valeur 4 timesteps avant (1h)
        → Capture l'évolution : une SpO2 qui chute de 3% en 1h est alarmante

    Variabilité (fc_std_4, spo2_std_4) :
        Écart-type sur 4 timesteps (1h glissante)
        → Une FC très variable = instabilité hémodynamique
    """
    log.info("🔧 Feature engineering...")
    df = df.sort_values([ENCOUNTER_ID_COL, TIME_COL]).copy()

    # MAP
    if "tas" in df.columns and "tad" in df.columns:
        df["map"] = df["tad"] + (df["tas"] - df["tad"]) / 3

    # Shock Index
    if "fc" in df.columns and "tas" in df.columns:
        df["shock_index"] = df["fc"] / df["tas"].replace(0, np.nan)
        df["shock_index"] = df["shock_index"].clip(0, 5)  # cap à 5 (valeurs absurdes)

    # Pulse Pressure
    if "tas" in df.columns and "tad" in df.columns:
        df["pulse_pressure"] = df["tas"] - df["tad"]

    # Tendances et variabilité (calculées par patient)
    def _rolling_features(g: pd.DataFrame) -> pd.DataFrame:
        for col in ["fc", "spo2", "tas"]:
            if col in g.columns:
                g[f"{col}_trend"] = g[col].diff(periods=4)       # delta sur 1h
                g[f"{col}_std_4"] = g[col].rolling(4, min_periods=2).std()
        return g

    df = df.groupby(ENCOUNTER_ID_COL, group_keys=False).apply(_rolling_features)
    log.info("  Features dérivées créées : map, shock_index, pulse_pressure, "
             "tendances × 3, std × 2")
    return df


# ─────────────────────────────────────────────
# 4. Labellisation
# ─────────────────────────────────────────────

def _is_deterioration_event(df_patient: pd.DataFrame) -> pd.Series:
    """
    Définit un événement de détérioration pour un patient.

    Critères (inspirés des scores NEWS2 / SOFA) :
        Un instant t est un événement si l'UN des critères suivants est rempli :
        - SpO2 < 88% (hypoxémie sévère)
        - FC > 130 ou FC < 40 (arythmie sévère)
        - TAS < 80 mmHg (hypotension sévère = critère de choc)
        - FR > 30 /min (détresse respiratoire)
        - Température > 39.5°C ou < 35°C (hyper/hypothermie sévère)
        - Lactate > 4 mmol/L (hyperlactatémie = hypoperfusion)
        - Shock Index > 1.2

    Pourquoi cette définition composite ?
        - Un seul critère évite les faux négatifs (1 signe = déjà grave)
        - Cohérent avec la littérature sur les scores d'alerte précoce (EWS)
    """
    s = pd.Series(False, index=df_patient.index)

    crit = {
        "spo2":        lambda x: x < 88,
        "fc":          lambda x: (x > 130) | (x < 40),
        "tas":         lambda x: x < 80,
        "fr":          lambda x: x > 30,
        "temperature": lambda x: (x > 39.5) | (x < 35),
        "lactate":     lambda x: x > 4,
        "shock_index": lambda x: x > 1.2,
    }
    for col, condition in crit.items():
        if col in df_patient.columns:
            s = s | condition(df_patient[col]).fillna(False)
    return s


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label = 1 si une détérioration survient dans les PREDICTION_HORIZON_H
    suivant le timestep t, 0 sinon.

    Logique par patient :
        1. On détecte les instants de détérioration (is_event)
        2. Pour chaque timestep t, on regarde si is_event est True
           dans la fenêtre [t+1, t+H] (H = 24 timesteps = 6h)
        3. On n'inclut pas l'instant t lui-même dans le look-ahead
           (sinon on "triche" en labellisant un état déjà dégradé)
    """
    log.info("🏷️  Labellisation (détérioration dans les 6h suivantes)...")
    horizon = PREDICTION_HORIZON_H * 4  # 6h × 4 timesteps/h = 24 timesteps

    def _label_patient(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(TIME_COL).copy()
        is_event = _is_deterioration_event(g)
        # Label = True si un événement existe dans les `horizon` timesteps suivants
        g[LABEL_COL] = [
            is_event.iloc[i+1: i+1+horizon].any()
            for i in range(len(g))
        ]
        g[LABEL_COL] = g[LABEL_COL].astype(int)
        return g

    df = df.groupby(ENCOUNTER_ID_COL, group_keys=False).apply(_label_patient)

    pos_rate = df[LABEL_COL].mean() * 100
    log.info(f"  Taux de positifs : {pos_rate:.1f}% "
             f"({df[LABEL_COL].sum():,} / {len(df):,})")
    return df


# ─────────────────────────────────────────────
# 5. Imputation KNN
# ─────────────────────────────────────────────

def impute_missing(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    k: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, KNNImputer]:
    """
    KNNImputer fit sur train seulement → transformé sur val et test.
    Règle d'or du ML : jamais de fuite d'information du val/test vers le train.

    k=5 : compromis entre précision et vitesse. Avec des séries temporelles
    médicales, 5 voisins capturent bien la diversité des profils patients.
    """
    log.info("🔧 Imputation des valeurs manquantes (Simple Médiane)...")

    imputer = SimpleImputer(strategy='median')
    # On entraîne l'imputer uniquement sur le TRAIN pour éviter la fuite de données
    df_train[feature_cols] = imputer.fit_transform(df_train[feature_cols])
    # On applique la même médiane au VAL et TEST
    df_val[feature_cols] = imputer.transform(df_val[feature_cols])
    df_test[feature_cols] = imputer.transform(df_test[feature_cols])

    log.info("  Imputation terminée — plus aucun NaN dans les features")
    return df_train, df_val, df_test, imputer


# ─────────────────────────────────────────────
# 6. Construction des fenêtres glissantes LSTM
# ─────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = WINDOW_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforme le DataFrame en tenseur 3D pour le LSTM.

    Sortie :
        X : (n_sequences, window_size, n_features)  → float32
        y : (n_sequences,)                          → int (0 ou 1)

    Stratégie :
        Pour chaque patient, on glisse une fenêtre de `window_size` timesteps.
        Le label de la fenêtre = label du DERNIER timestep de la fenêtre
        (c'est à partir de ce moment qu'on prédit la détérioration future).

    Pas de chevauchement entre patients :
        On repart à zéro à chaque nouveau patient (pas de fenêtre
        qui chevauche deux séjours différents).

    Exigence : au moins window_size+1 timesteps par patient,
    sinon le patient est ignoré.
    """
    log.info(f"📦 Construction des séquences ({window_size} timesteps × "
             f"{len(feature_cols)} features)...")

    X_list, y_list = [], []
    patients = df[ENCOUNTER_ID_COL].unique()
    skipped = 0

    for pid in patients:
        patient_df = df[df[ENCOUNTER_ID_COL] == pid].sort_values(TIME_COL)
        vals = patient_df[feature_cols].values.astype(np.float32)
        labels = patient_df[LABEL_COL].values.astype(np.int32)

        n = len(vals)
        if n < window_size + 1:
            skipped += 1
            continue

        # Fenêtre glissante
        for i in range(n - window_size):
            X_list.append(vals[i: i + window_size])
            y_list.append(labels[i + window_size])  # label au dernier step

    if not X_list:
        raise ValueError("Aucune séquence construite — vérifie les données d'entrée.")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)

    log.info(f"  {len(X):,} séquences construites "
             f"({skipped} patients ignorés < {window_size+1} timesteps)")
    log.info(f"  Shape X : {X.shape}, y : {y.shape}")
    log.info(f"  Taux de positifs dans ce split : {y.mean()*100:.1f}%")
    return X, y


# ─────────────────────────────────────────────
# 7. Normalisation
# ─────────────────────────────────────────────

def normalize(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    StandardScaler fit sur X_train uniquement.
    Les données sont en forme 3D (samples, timesteps, features).
    On reshape en 2D pour le scaler, puis on remet en 3D.

    Pourquoi StandardScaler et pas MinMax ?
        → MinMax est sensible aux outliers résiduels.
        → StandardScaler (µ=0, σ=1) est plus robuste pour les LSTM.
    """
    log.info("📏 Normalisation (StandardScaler)...")
    n_features = X_train.shape[2]

    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    log.info(f"  Moyenne train (feature 0) après scale : "
             f"{X_train[:, :, 0].mean():.4f} ≈ 0 ✓")
    return X_train, X_val, X_test, scaler


# ─────────────────────────────────────────────
# 8. Split patient-aware
# ─────────────────────────────────────────────

def split_by_patient(
    df: pd.DataFrame,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split train/val/test en groupant par patient.

    CRITIQUE : on ne coupe JAMAIS un patient en deux.
    Si on mettait les premières 6h d'un patient en train et les 6h suivantes
    en val, le modèle "verrait" déjà ce patient → biais optimiste sur la val.

    GroupShuffleSplit de sklearn fait ça proprement.
    """
    log.info("✂️  Split train/val/test (patient-aware)...")
    patients = df[ENCOUNTER_ID_COL].values
    indices  = np.arange(len(df))

    # Test split
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state
    )
    train_val_idx, test_idx = next(gss_test.split(indices, groups=patients))

    # Val split (sur la partie train_val)
    val_ratio_adjusted = val_ratio / (1 - test_ratio)
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_ratio_adjusted, random_state=random_state
    )
    train_idx, val_idx = next(
        gss_val.split(train_val_idx, groups=patients[train_val_idx])
    )
    train_idx = train_val_idx[train_idx]
    val_idx   = train_val_idx[val_idx]

    df_train = df.iloc[train_idx].copy()
    df_val   = df.iloc[val_idx].copy()
    df_test  = df.iloc[test_idx].copy()

    def _log_split(name, d):
        n_pat = d[ENCOUNTER_ID_COL].nunique()
        pos = d[LABEL_COL].mean() * 100 if LABEL_COL in d.columns else 0
        log.info(f"  {name:6s}: {len(d):>8,} lignes | {n_pat:>5,} patients | "
                 f"{pos:.1f}% positifs")

    _log_split("TRAIN", df_train)
    _log_split("VAL",   df_val)
    _log_split("TEST",  df_test)

    return df_train, df_val, df_test


# ─────────────────────────────────────────────
# 9. Calcul des class weights
# ─────────────────────────────────────────────

def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Corrige le déséquilibre de classes (~15% positifs).
    Formule sklearn : weight_i = n_total / (n_classes × n_i)

    Résultat typique :
        {0: 0.59, 1: 3.33}
    → Le modèle "penalise" 3.3× plus les erreurs sur les positifs.
    → Essentiel pour ne pas avoir un recall de 0% sur les détériorations.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    log.info(f"⚖️  Class weights : {cw}")
    return cw


# ─────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────

def run_preprocessing_pipeline(
    df_raw: pd.DataFrame,
    output_dir: str = "data/splits",
    models_dir: str = "models",
) -> Dict:
    """
    Exécute toutes les étapes de prétraitement dans l'ordre et sauvegarde
    les arrays numpy prêts pour l'entraînement.

    Retourne un dictionnaire contenant :
        - X_train, y_train
        - X_val, y_val
        - X_test, y_test
        - scaler, imputer, class_weights
        - feature_cols
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("🚀 PIPELINE DE PRÉTRAITEMENT EARLYALERT")
    log.info("=" * 60)

    # ── Étape 1 : Filtrage aberrants
    df = filter_outliers(df_raw.copy())

    # ── Étape 2 : Rééchantillonnage
    df = resample_all(df)

    # ── Étape 3 : Feature engineering
    df = engineer_features(df)

    # ── Étape 4 : Labellisation
    df = create_labels(df)

    # ── Vérification des features disponibles
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        log.warning(f"  Features manquantes (ignorées) : {missing_features}")
    log.info(f"  Features utilisées ({len(available_features)}) : {available_features}")

    # ── Étape 5 : Split patient-aware
    df_train, df_val, df_test = split_by_patient(df)

    # ── Étape 6 : Imputation KNN (fit sur train)
    df_train, df_val, df_test, imputer = impute_missing(
        df_train, df_val, df_test, available_features
    )

    # ── Étape 7 : Séquences
    log.info("\n📦 Séquences TRAIN")
    X_train, y_train = build_sequences(df_train, available_features)
    log.info("📦 Séquences VAL")
    X_val,   y_val   = build_sequences(df_val,   available_features)
    log.info("📦 Séquences TEST")
    X_test,  y_test  = build_sequences(df_test,  available_features)

    # ── Étape 8 : Normalisation (fit sur train)
    X_train, X_val, X_test, scaler = normalize(X_train, X_val, X_test)

    # ── Étape 9 : Class weights
    class_weights = compute_class_weights(y_train)

    # ── Sauvegarde arrays
    log.info("\n💾 Sauvegarde des arrays...")
    np.save(out_path / "X_train.npy", X_train)
    np.save(out_path / "y_train.npy", y_train)
    np.save(out_path / "X_val.npy",   X_val)
    np.save(out_path / "y_val.npy",   y_val)
    np.save(out_path / "X_test.npy",  X_test)
    np.save(out_path / "y_test.npy",  y_test)
    np.save(out_path / "class_weights.npy", class_weights)

    # Sauvegarde scaler et imputer
    joblib.dump(scaler,  Path(models_dir) / "scaler.pkl")
    joblib.dump(imputer, Path(models_dir) / "imputer.pkl")

    # Sauvegarde feature list
    import json
    with open(Path(models_dir) / "feature_cols.json", "w") as f:
        json.dump(available_features, f)

    log.info("=" * 60)
    log.info("✅ PREPROCESSING TERMINÉ")
    log.info(f"   X_train : {X_train.shape}")
    log.info(f"   X_val   : {X_val.shape}")
    log.info(f"   X_test  : {X_test.shape}")
    log.info(f"   Features : {len(available_features)}")
    log.info("=" * 60)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "scaler":  scaler,
        "imputer": imputer,
        "class_weights": class_weights,
        "feature_cols":  available_features,
    }


if __name__ == "__main__":
    # Charge le fichier mergé produit par data_loader.py
    raw_path = Path("data/processed/df_raw_merged.parquet")
    if not raw_path.exists():
        raise FileNotFoundError(
            "Lance d'abord data_loader.py pour générer df_raw_merged.parquet"
        )

    log.info(f"📂 Chargement de {raw_path}...")
    df_raw = pd.read_parquet(raw_path)
    log.info(f"   {df_raw.shape[0]:,} lignes × {df_raw.shape[1]} colonnes")

    results = run_preprocessing_pipeline(df_raw)

    print("\n✅ Tout est prêt. Lance maintenant train.py !")
    print(f"   X_train shape : {results['X_train'].shape}")
    print(f"   Class weights : {results['class_weights']}")

