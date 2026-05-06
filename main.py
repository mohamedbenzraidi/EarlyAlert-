from src.preprocessing import nettoyageChartevents
from src.preprocessing import build_final_dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CHARTEVENTS_CLEAN = os.path.join(BASE_DIR, "data", "processed", "chartevents.csv")
PATH_LABEVENTS_CLEAN = os.path.join(BASE_DIR, "data", "processed", "labevents.csv")
PATH_CHARTEVENTS_RAW = os.path.join(BASE_DIR, "data", "raw", "chartevents.csv")
PATH_LABEVENTS_RAW = os.path.join(BASE_DIR, "data", "raw", "labevents.csv")
PATH_PATIENTS_RAW = os.path.join(BASE_DIR, "data", "raw", "patients.csv")
PATH_ICUSTAYS_RAW = os.path.join(BASE_DIR, "data", "raw", "icustays.csv")

conditions1 = {
    "HR": (20, 220),
    "SpO2": (50, 100),
    "SBP": (40, 250),
    "DBP": (30, 150),
    "Resp": (4, 60),
    "Temp": (30, 45)
}

conditions2 = {
    "Lactate": (0, 20),
    "WBC": (0, 100),
    "Creatinine": (0, 20),
}

nettoyageChartevents(PATH_CHARTEVENTS_RAW, os.path.join(BASE_DIR, "data", "processed", "chartevents.csv"), conditions1)
nettoyageChartevents(PATH_LABEVENTS_RAW, os.path.join(BASE_DIR, "data", "processed", "labevents.csv"), conditions2)

build_final_dataset(
    PATH_CHARTEVENTS_CLEAN,
    PATH_LABEVENTS_CLEAN,
    PATH_PATIENTS_RAW,
    PATH_ICUSTAYS_RAW,
    os.path.join(BASE_DIR, "data", "processed", "merged_dataset.csv")
)

