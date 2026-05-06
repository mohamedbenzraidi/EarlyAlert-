import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import os

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

    # Utiliser group_keys=False pour éviter de créer des index dupliqués
    df = df.groupby("subject_id", group_keys=False).apply(lambda x: x.ffill().bfill())
    df = df.reset_index(drop=True)

    # =========================
    # 10. FEATURES DERIVEES
    # =========================

    # Shock Index
    if "HR" in df.columns and "SBP" in df.columns:
        df["shock_index"] = df["HR"] / df["SBP"]

    # Trends
    if "HR" in df.columns:
        df["HR_trend"] = df.groupby("subject_id")["HR"].diff()

    if "SpO2" in df.columns:
        df["SpO2_trend"] = df.groupby("subject_id")["SpO2"].diff()

    # Variabilité
    if "HR" in df.columns:
        df["HR_var"] = (
            df.groupby("subject_id")["HR"]
              .rolling(window=4)
              .std()
              .reset_index(level=0, drop=True)
        )

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


def prepare_data(path,output_dir) :
    df = pd.read_csv(path)

    #labelisation si un patient est en deterioration en 6h ou non
    df["label"] = 0  # Initialiser la colonne label à 0
    df["deterioration"] = df["shock_index"] > 1

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

    #features

    features = [
        "HR", "SBP", "DBP", "SpO2", "Temp","Resp",
        "Creatinine", "Lactate", "WBC",
        "shock_index", "HR_trend", "SpO2_trend", "HR_var"
    ]

    #normalisation
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    #fenetres glissantes
    X,y,groups = [],[],[]
    for stay_id, group in df.groupby("stay_id"):
        group = group.reset_index(drop=True)

        for i in range(len(group) - window):
            seq = group.loc[i:i + window - 1, features].values
            label = group.loc[i + window - 1, "label"]

            X.append(seq)
            y.append(label)
            groups.append(stay_id)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    print("X shape:", X.shape)
    print("y mean (positifs):", np.mean(y))

    #split
    unique_stays = np.unique(groups)

    train_ids, temp_ids = train_test_split(unique_stays, test_size=0.30, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)

    train_mask = np.isin(groups, train_ids)
    val_mask = np.isin(groups, val_ids)
    test_mask = np.isin(groups, test_ids)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

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

    # sauvegarde des données
    np.save(os.path.join(output_dir,"X_train.npy"), X_train)
    np.save(os.path.join(output_dir,"y_train.npy"), y_train)

    np.save(os.path.join(output_dir,"X_val.npy"), X_val)
    np.save(os.path.join(output_dir,"y_val.npy"), y_val)

    np.save(os.path.join(output_dir,"X_test.npy"), X_test)
    np.save(os.path.join(output_dir,"y_test.npy"), y_test)

    # class weights
    np.save(os.path.join(output_dir,"class_weight.npy"), class_weight)


