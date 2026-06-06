"""
EarlyAlert — Pipeline Complet
Data Loading → Preprocessing → Training → Evaluation
"""

import os
import sys
import pandas as pd
import numpy as np  # <--- CORRIGÉ : Import indispensable pour l'échantillonnage
import gc           # <--- AJOUTÉ : Pour vider la RAM entre les étapes

from src.data_loader import CLIFDataLoader
from src.preprocessing import run_preprocessing_pipeline
from src.train import train_model
from src.evaluate import evaluate_model

# ═══════════════════════════════════════════════════════════════════════════
# Configuration des chemins
# ═══════════════════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins pour data_loader
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# S'assurer que les dossiers existent
for folder in [PROCESSED_DIR, SPLITS_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("🏥 EARLYALERT — PIPELINE COMPLET")
    print("=" * 70)

    # ───────────────────────────────────────────────────────────────────────
    # 1️⃣ CHARGEMENT (Version optimisée : on lit le fichier déjà prêt)
    # ───────────────────────────────────────────────────────────────────────
    print("\n1️⃣ Chargement et fusion des données CLIF...")
    try:
        loader = CLIFDataLoader(raw_dir=RAW_DIR, min_icu_hours=6)
        df_raw = loader.load_all()
        print("   ✅ Données chargées et fusionnées")
    except Exception as e:
        print(f"   ❌ Erreur chargement: {e}")
        return False

    # Sauvegarde intermédiaire
    df_raw.to_parquet(os.path.join(PROCESSED_DIR, "df_raw_merged.parquet"), index=False)
    print(f"   💾 Sauvegardé dans {PROCESSED_DIR}/df_raw_merged.parquet")

    # ───────────────────────────────────────────────────────────────────────
    # 2️⃣ PRÉTRAITEMENT (filtrage, rééchantillonnage, features, etc.)
    # ───────────────────────────────────────────────────────────────────────
    print("\n2️⃣ Prétraitement des données...")
    try:
        # Harmonisation du nom de la colonne ID
        if 'hospitalization_id' in df_raw.columns:
            df_raw = df_raw.rename(columns={'hospitalization_id': 'encounter_id'})

        # --- LIMITATION DE LA CHARGE POUR 16Go RAM ---
        n_patients = 25000
        all_patients = df_raw['encounter_id'].unique()

        if len(all_patients) > n_patients:
            print(f"   ⚠️ Trop de données ({len(all_patients)} patients). Réduction à {n_patients}...")
            # Choix aléatoire reproductible avec random_state si tu veux
            selected_patients = np.random.choice(all_patients, n_patients, replace=False)
            df_raw = df_raw[df_raw['encounter_id'].isin(selected_patients)].copy()

            # Nettoyage mémoire immédiat
            gc.collect()
        # --------------------------------------------

        results = run_preprocessing_pipeline(
            df_raw,
            output_dir=SPLITS_DIR,
            models_dir=MODELS_DIR
        )

        # Très important : Libérer df_raw une fois le preprocessing fini pour le Train
        del df_raw
        gc.collect()

        print("   ✅ Prétraitement terminé")
    except Exception as e:
        print(f"   ❌ Erreur prétraitement: {e}")
        return False

    # ───────────────────────────────────────────────────────────────────────
    # 3️⃣ ENTRAÎNEMENT DU MODÈLE
    # ───────────────────────────────────────────────────────────────────────
    print("\n3️⃣ Entraînement du modèle LSTM + Attention...")
    print(f"   🚀 Utilisation de la RTX 4060 détectée (si drivers OK)")
    try:
        train_model()
        print("   ✅ Modèle entraîné et sauvegardé")
    except Exception as e:
        print(f"   ❌ Erreur entraînement: {e}")
        return False

    # ───────────────────────────────────────────────────────────────────────
    # 4️⃣ ÉVALUATION DU MODÈLE
    # ───────────────────────────────────────────────────────────────────────
    print("\n4️⃣ Évaluation du modèle...")
    try:
        evaluate_model()
        print("   ✅ Évaluation terminée")
    except Exception as e:
        print(f"   ❌ Erreur évaluation: {e}")
        return False

    # ───────────────────────────────────────────────────────────────────────
    # ✅ RÉSUMÉ FINAL
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS !")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)