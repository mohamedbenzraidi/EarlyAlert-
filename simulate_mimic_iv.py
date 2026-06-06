# simulate_mimic_iv.py

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)

# ===============================
# 📁 CREATE FOLDERS
# ===============================
os.makedirs("data/raw", exist_ok=True)

# ===============================
# 👥 PARAMETERS
# ===============================
N_PATIENTS = 100
MAX_HOURS = 48  # durée max séjour ICU

# ===============================
# 🧍 PATIENTS TABLE
# ===============================
patients = []

for pid in range(N_PATIENTS):
    patients.append({
        "subject_id": pid,
        "gender": np.random.choice(["M", "F"]),
        "anchor_age": np.random.randint(18, 90)
    })

patients_df = pd.DataFrame(patients)
patients_df.to_csv("data/raw/patients.csv", index=False)

# ===============================
# 🏥 ICU STAYS
# ===============================
icustays = []

for pid in range(N_PATIENTS):
    intime = datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 10))
    length_hours = np.random.randint(12, MAX_HOURS)

    icustays.append({
        "subject_id": pid,
        "stay_id": pid,
        "intime": intime,
        "outtime": intime + timedelta(hours=length_hours)
    })

icustays_df = pd.DataFrame(icustays)
icustays_df.to_csv("data/raw/icustays.csv", index=False)

# ===============================
# ❤️ CHARTEVENTS (vital signs)
# ===============================
variables = {
    "HR": (60, 100),
    "SpO2": (92, 100),
    "SBP": (100, 140),
    "DBP": (60, 90),
    "Resp": (12, 20),
    "Temp": (36.5, 37.5)
}

chartevents = []

for _, stay in icustays_df.iterrows():
    pid = stay["subject_id"]
    current_time = stay["intime"]
    end_time = stay["outtime"]

    while current_time < end_time:
        for var, (low, high) in variables.items():

            # valeurs normales
            value = np.random.normal((low + high) / 2, 5)

            # anomalies médicales réalistes
            if var == "HR" and np.random.rand() < 0.1:
                value = np.random.normal(130, 10)  # tachycardie

            if var == "SpO2" and np.random.rand() < 0.1:
                value = np.random.normal(85, 5)  # hypoxie

            if var == "SBP" and np.random.rand() < 0.1:
                value = np.random.normal(85, 10)  # hypotension

            # valeurs manquantes
            if np.random.rand() < 0.2:
                value = np.nan

            chartevents.append({
                "subject_id": pid,
                "stay_id": pid,
                "charttime": current_time,
                "itemid": var,
                "valuenum": value
            })

        # temps irrégulier (5 à 60 min)
        current_time += timedelta(minutes=np.random.randint(5, 60))

chartevents_df = pd.DataFrame(chartevents)
chartevents_df.to_csv("data/raw/chartevents.csv", index=False)

# ===============================
# 🧪 LABEVENTS (biologie)
# ===============================
lab_vars = {
    "Lactate": (0.5, 2.0),
    "WBC": (4, 11),
    "Creatinine": (0.6, 1.3)
}

labevents = []

for _, stay in icustays_df.iterrows():
    pid = stay["subject_id"]
    current_time = stay["intime"]
    end_time = stay["outtime"]

    while current_time < end_time:

        for var, (low, high) in lab_vars.items():

            value = np.random.normal((low + high) / 2, 0.5)

            # anomalies critiques
            if var == "Lactate" and np.random.rand() < 0.1:
                value = np.random.normal(4, 1)  # choc

            if np.random.rand() < 0.3:
                value = np.nan

            labevents.append({
                "subject_id": pid,
                "stay_id": pid,
                "charttime": current_time,
                "itemid": var,
                "valuenum": value
            })

        current_time += timedelta(hours=4)

labevents_df = pd.DataFrame(labevents)
labevents_df.to_csv("data/raw/labevents.csv", index=False)

print("✅ Simulation MIMIC-IV terminée !")
print("📁 Données générées dans : data/raw/")