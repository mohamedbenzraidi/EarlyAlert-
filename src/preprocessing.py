import json
import os

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

def nettoyageChartevents(path_raw, path_cleaned,conditions):
    data = pd.read_csv(path_raw)
    df = pd.DataFrame(data)

    # conditions1 = {
    #     "HR":   (20, 220),
    #     "SpO2": (50, 100),
    #     "SBP":  (40, 250),
    #     "DBP":  (30, 150),
    #     "Resp": (4, 60),
    #     "Temp": (30, 45)
    # }

    # conditions2 ={
    #     "Lactate": (0, 20),
    #     "WBC": (0, 100),
    #     "Creatinine": (0, 20),
    # }

    def est_valide(row):
        item = row['itemid']
        val = row['valuenum']
        if pd.isna(val) or item not in conditions:
            return False
        low, high = conditions[item]
        return low <= val <= high

    # Filtrage des lignes
    df_filtre = df[df.apply(est_valide, axis=1)].copy()

    df_pivoted = df_filtre.pivot_table(
        index=['subject_id', 'charttime'],
        columns='itemid',
        values='valuenum'
    ).reset_index()

    vitals = list(conditions.keys())
    df_pivoted.to_csv(path_cleaned, index=False)

    print("Nettoyage terminé et sauvegarde avec succès. Chartevents")


def build_final_dataset(
    path_chartevents,
    path_labevents,
    path_patients,
    path_icustays,
    output_path
):
    # =========================
    # 1. LOAD DATA
    # =========================
    chartevents = pd.read_csv(path_chartevents)
    labevents = pd.read_csv(path_labevents)
    patients = pd.read_csv(path_patients)
    icustays = pd.read_csv(path_icustays)

    # =========================
    # 2. FIX DATETIME
    # =========================
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
    labevents["charttime"] = pd.to_datetime(labevents["charttime"])

    # =========================
    # 3. CONCAT (AU LIEU DE MERGE ❗)
    # =========================
    df = pd.concat([chartevents, labevents], ignore_index=True)

    # =========================
    # 4. SORT
    # =========================
    df = df.sort_values(["subject_id", "charttime"])

    # =========================
    # 5. RESAMPLING (15 min)
    # =========================
    df = (
        df.set_index("charttime")
        .groupby("subject_id")
        .resample("15min")
        .mean(numeric_only=True)
    )

    # Si subject_id est à la fois dans l'index et dans les colonnes,
    # on supprime la colonne avant le reset_index
    if "subject_id" in df.columns:
        df = df.drop(columns=["subject_id"])

    df = df.reset_index()

    # =========================
    # 6. MERGE PATIENTS
    # =========================
    df = df.merge(patients, on="subject_id", how="left")

    # =========================
    # 7. MERGE ICUSTAYS
    # =========================
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    df = df.merge(icustays, on="subject_id", how="left")

    # =========================
    # 8. FILTRER PAR SEJOUR ICU
    # =========================
    df = df[
        (df["charttime"] >= df["intime"]) &
        (df["charttime"] <= df["outtime"])
    ]

    # =========================
    # 9. IMPUTATION (IMPORTANT)
    # =========================
    df = df.sort_values(["subject_id", "charttime"])
    # Par séjour ICU si disponible (évite de mélanger plusieurs stays d'un même patient)
    _grp_stay = "stay_id" if "stay_id" in df.columns else "subject_id"
    df = df.groupby(_grp_stay, group_keys=False).apply(lambda x: x.ffill())
    df = df.reset_index(drop=True)

    # =========================
    # 10. FEATURES DERIVEES
    # =========================

    # Shock Index (éviter division par zéro → inf)
    if "HR" in df.columns and "SBP" in df.columns:
        sbp_safe = df["SBP"].replace(0, np.nan)
        df["shock_index"] = df["HR"] / sbp_safe

    _g = _grp_stay
    # Trends
    if "HR" in df.columns:
        df["HR_trend"] = df.groupby(_g)["HR"].diff()

    if "SpO2" in df.columns:
        df["SpO2_trend"] = df.groupby(_g)["SpO2"].diff()

    # Variabilité
    if "HR" in df.columns:
        df["HR_var"] = (
            df.groupby(_g)["HR"]
              .rolling(window=4)
              .std()
              .reset_index(level=0, drop=True)
        )

    df = df.replace([np.inf, -np.inf], np.nan)

    # =========================
    # 11. DROP colonnes inutiles
    # =========================
    df = df.drop(columns=["intime", "outtime"], errors="ignore")

    # =========================
    # 12. SAVE
    # =========================
    df.to_csv(output_path, index=False)

    print(" Dataset final créé avec succès :", output_path)

    return df


def _impute_feature_matrix(df, features, group_key="stay_id"):
    """Remplace inf/NaN pour que StandardScaler et le réseau ne reçoivent pas de NaN."""
    df = df.replace([np.inf, -np.inf], np.nan)
    present = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if missing:
        print("Colonnes absentes (ignorées):", missing)
    if not present:
        raise ValueError("Aucune colonne de features présente dans le CSV fusionné.")

    if group_key in df.columns:
        df = df.sort_values([group_key, "charttime"] if "charttime" in df.columns else [group_key])
        for c in present:
            df[c] = df.groupby(group_key, group_keys=False)[c].transform(
                lambda s: s.ffill().bfill()
            )
    for c in present:
        med = df[c].median()
        if pd.isna(med):
            med = 0.0
        df[c] = df[c].fillna(med)

    n_bad = int(df[present].isna().sum().sum())
    if n_bad > 0:
        print("Attention: NaN résiduels après imputation:", n_bad, "→ remplissage 0.")
        df[present] = df[present].fillna(0.0)
    return df, present


def prepare_data(path,output_dir) :
    df = pd.read_csv(path)
    if "charttime" in df.columns:
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

    features = [
        "HR", "SBP", "DBP", "SpO2", "Temp","Resp",
        "Creatinine", "Lactate", "WBC",
        "shock_index", "HR_trend", "SpO2_trend", "HR_var"
    ]

    #labelisation si un patient est en deterioration en 6h ou non
    df["label"] = 0  # Initialiser la colonne label à 0
    if "shock_index" in df.columns:
        stats = df["shock_index"].describe()
        print(f"--- STATS SHOCK INDEX ---")
        print(stats)
        # Si max > 5 ou mean est étrange, on a un problème
        if df["shock_index"].max() > 10:
            print("ATTENTION: Valeurs aberrantes détectées dans shock_index !")
    # Événement « détérioration » à t : critères cliniques combinés (données brutes avant imputation)
    ev_si = (df["shock_index"] > 1.0) & df["shock_index"].notna()
    ev = ev_si
    if "Lactate" in df.columns:
        ev = ev | (df["Lactate"] > 2.2).fillna(False)
    if "SpO2" in df.columns:
        ev = ev | (df["SpO2"] < 88).fillna(False)
    if "Resp" in df.columns:
        ev = ev | (df["Resp"] > 32).fillna(False)
    df["deterioration"] = ev

    window = 24 #6h

    def apply_label(group):
        group["label"] = (group["deterioration"].shift(-window)
                          .rolling(window,min_periods=1)
                          .max()
                          .shift(-window+1)
                          .fillna(0)
                          )
        return group

    df = df.groupby("stay_id", group_keys=False).apply(apply_label)

    df = df.drop(columns=["deterioration"], errors="ignore")

    df, features = _impute_feature_matrix(df, features, group_key="stay_id")

    # Fenêtres sur données brutes (pas encore scalées) — évite la fuite du StandardScaler
    X, y, groups = [], [], []
    for stay_id, group in df.groupby("stay_id"):
        group = group.reset_index(drop=True)
        raw = group[features].astype(np.float64).values
        lab = group["label"].values
        step = 4
        for i in range(len(group) - window -1 , step):
            X.append(raw[i : i + window])
            y.append(float(lab[i + window]))
            groups.append(stay_id)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    groups = np.array(groups)

    print("X shape (brut):", X.shape)
    print("y mean (positifs):", np.mean(y))

    unique_stays = np.unique(groups)
    train_ids, temp_ids = train_test_split(unique_stays, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    train_mask = np.isin(groups, train_ids)
    val_mask = np.isin(groups, val_ids)
    test_mask = np.isin(groups, test_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    n_feat = len(features)
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape).astype(np.float32)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape).astype(np.float32)
    X_test = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape).astype(np.float32)

    joblib.dump(scaler, os.path.join(output_dir, "feature_scaler.joblib"))
    with open(os.path.join(output_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2)
    print("Scaler fit sur TRAIN uniquement — sauvegardé:", os.path.join(output_dir, "feature_scaler.joblib"))

    print("Train:", X_train.shape)
    print("Val:", X_val.shape)
    print("Test:", X_test.shape)

    # -----------------------------
    # 6. CLASS IMBALANCE
    # -----------------------------
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    print("Class weights:", class_weight)

    os.makedirs("data/splits", exist_ok=True)

    unique_stays = np.unique(groups)
    print(f"--- DIAGNOSTIC DATASET ---")
    print(f"Nombre total de séjours ICU (Stays): {len(unique_stays)}")
    print(f"Nombre total de fenêtres générées: {len(X)}")
    print(f"Ratio fenêtres/stay: {len(X) / len(unique_stays):.2f}")

    # sauvegarde des données
    np.save(os.path.join(output_dir,"X_train.npy"), X_train)
    np.save(os.path.join(output_dir,"y_train.npy"), y_train)

    np.save(os.path.join(output_dir,"X_val.npy"), X_val)
    np.save(os.path.join(output_dir,"y_val.npy"), y_val)

    np.save(os.path.join(output_dir,"X_test.npy"), X_test)
    np.save(os.path.join(output_dir,"y_test.npy"), y_test)

    # class weights
    np.save(os.path.join(output_dir,"class_weight.npy"), class_weight)


