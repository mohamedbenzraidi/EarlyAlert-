"""
EarlyAlert — Pipeline Complet
Nettoyage → Fusion → Séquences → Normalisation → Split → Entraînement
"""

from src.preprocessing import nettoyageChartevents, build_final_dataset, prepare_data
from src.train import train_model
import os
import sys

# ═══════════════════════════════════════════════════════════════════════════
# Configuration des chemins
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins brutes (RAW)
PATH_CHARTEVENTS_RAW = os.path.join(BASE_DIR, "data", "raw", "chartevents.csv")
PATH_LABEVENTS_RAW = os.path.join(BASE_DIR, "data", "raw", "labevents.csv")
PATH_PATIENTS_RAW = os.path.join(BASE_DIR, "data", "raw", "patients.csv")
PATH_ICUSTAYS_RAW = os.path.join(BASE_DIR, "data", "raw", "icustays.csv")

# Chemins nettoyés
PATH_CHARTEVENTS_CLEAN = os.path.join(BASE_DIR, "data", "processed", "chartevents.csv")
PATH_LABEVENTS_CLEAN = os.path.join(BASE_DIR, "data", "processed", "labevents.csv")
PATH_MERGED_DATASET = os.path.join(BASE_DIR, "data", "processed", "merged_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")

# Conditions de filtrage (valeurs aberrantes)
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

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    
    print("\n" + "="*70)
    print("🏥 EARLYALERT — PIPELINE COMPLET")
    print("="*70)
    
    # ───────────────────────────────────────────────────────────────────────
    # 1️⃣ NETTOYAGE CHARTEVENTS (constantes vitales)
    # ───────────────────────────────────────────────────────────────────────
    print("\n1️⃣ Nettoyage des constantes vitales (chartevents)...")
    try:
        nettoyageChartevents(
            PATH_CHARTEVENTS_RAW,
            PATH_CHARTEVENTS_CLEAN,
            conditions1
        )
        print("   ✅ Chartevents nettoyé")
    except Exception as e:
        print(f"   ❌ Erreur chartevents: {e}")
        return False
    
    # ───────────────────────────────────────────────────────────────────────
    # 2️⃣ NETTOYAGE LABEVENTS (biomarqueurs)
    # ───────────────────────────────────────────────────────────────────────
    print("\n2️⃣ Nettoyage des biomarqueurs (labevents)...")
    try:
        nettoyageChartevents(
            PATH_LABEVENTS_RAW,
            PATH_LABEVENTS_CLEAN,
            conditions2
        )
        print("   ✅ Labevents nettoyé")
    except Exception as e:
        print(f"   ❌ Erreur labevents: {e}")
        return False
    
    # ───────────────────────────────────────────────────────────────────────
    # 3️⃣ FUSION DES DONNÉES
    # ───────────────────────────────────────────────────────────────────────
    print("\n3️⃣ Fusion des données (chartevents + labevents + patients + icustays)...")
    try:
        build_final_dataset(
            PATH_CHARTEVENTS_CLEAN,
            PATH_LABEVENTS_CLEAN,
            PATH_PATIENTS_RAW,
            PATH_ICUSTAYS_RAW,
            PATH_MERGED_DATASET
        )
        print("   ✅ Dataset fusionné créé")
    except Exception as e:
        print(f"   ❌ Erreur fusion: {e}")
        return False
    
    # ───────────────────────────────────────────────────────────────────────
    # 4️⃣ CRÉATION DES SÉQUENCES (24 timesteps × 13 features)
    # ───────────────────────────────────────────────────────────────────────
    print("\n4️⃣ Création des séquences (fenêtres glissantes 24×13)...")
    try:
        prepare_data(PATH_MERGED_DATASET, OUTPUT_DIR)
        print("   ✅ Séquences créées")
        print(f"   📁 Fichiers .npy générés dans {OUTPUT_DIR}/")
    except Exception as e:
        print(f"   ❌ Erreur séquences: {e}")
        return False
    
    # ───────────────────────────────────────────────────────────────────────
    # 5️⃣ ENTRAÎNEMENT DU MODÈLE
    # ───────────────────────────────────────────────────────────────────────
    print("\n5️⃣ Entraînement du modèle LSTM + Attention...")
    print("   ⏳ Cela peut prendre 20-30 min sur GPU...")
    try:
        train_model()
        print("   ✅ Modèle entraîné et validé")
    except Exception as e:
        print(f"   ❌ Erreur entraînement: {e}")
        return False
    
    # ───────────────────────────────────────────────────────────────────────
    # ✅ SUCCÈS
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS !")
    print("="*70)
    print("""
    📊 Fichiers générés:
    ├─ data/processed/chartevents.csv (nettoyé)
    ├─ data/processed/labevents.csv (nettoyé)
    ├─ data/processed/merged_dataset.csv (fusionné)
    ├─ data/processed/X_train.npy
    ├─ data/processed/y_train.npy
    ├─ data/processed/X_val.npy
    ├─ data/processed/y_val.npy
    ├─ data/processed/X_test.npy
    ├─ data/processed/y_test.npy
    ├─ data/processed/class_weight.npy
    └─ models/best_model.h5 (modèle entraîné)
    
    🚀 Prochaines étapes:
    ├─ python RUN_EXAMPLES.py          (faire des prédictions)
    ├─ streamlit run streamlit_app.py  (lancer le dashboard)
    └─ python -c "from src.evaluate import evaluate_model; evaluate_model()"
    """)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


