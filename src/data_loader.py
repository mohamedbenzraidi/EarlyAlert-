"""
data_loader.py — Chargement des données MIMIC-IV
Fonctions pour charger, valider et préparer les données brutes
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple


def load_raw_data(
    path_chartevents: str,
    path_labevents: str,
    path_patients: str,
    path_icustays: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les 4 tables brutes de MIMIC-IV
    
    Args:
        path_chartevents : Chemin vers chartevents.csv (constantes vitales)
        path_labevents   : Chemin vers labevents.csv (résultats biologiques)
        path_patients    : Chemin vers patients.csv (données démographiques)
        path_icustays    : Chemin vers icustays.csv (séjours ICU)
    
    Returns:
        Tuple de 4 DataFrames chargés
    
    Raises:
        FileNotFoundError : Si un fichier n'existe pas
    """
    print("📂 Chargement des données brutes MIMIC-IV...")
    
    if not all(os.path.exists(p) for p in [path_chartevents, path_labevents, 
                                            path_patients, path_icustays]):
        raise FileNotFoundError("❌ Un ou plusieurs fichiers .csv sont manquants dans data/raw/")
    
    chartevents = pd.read_csv(path_chartevents)
    labevents = pd.read_csv(path_labevents)
    patients = pd.read_csv(path_patients)
    icustays = pd.read_csv(path_icustays)
    
    print(f"✅ Chartevents : {chartevents.shape[0]:,} lignes")
    print(f"✅ Labevents : {labevents.shape[0]:,} lignes")
    print(f"✅ Patients : {patients.shape[0]:,} patients")
    print(f"✅ ICU Stays : {icustays.shape[0]:,} séjours")
    
    return chartevents, labevents, patients, icustays


def load_processed_data(
    output_dir: str = "data/processed"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
           np.ndarray, np.ndarray, dict]:
    """
    Charge les données pré-traitées (après preprocessing)
    Retourne les matrices d'entraînement, validation et test
    
    Args:
        output_dir : Dossier contenant les fichiers .npy
    
    Returns:
        Tuple (X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
    
    Raises:
        FileNotFoundError : Si les fichiers .npy n'existent pas
    """
    print("\n📦 Chargement des données pré-traitées...")
    
    required_files = [
        "X_train.npy", "y_train.npy",
        "X_val.npy", "y_val.npy",
        "X_test.npy", "y_test.npy",
        "class_weight.npy"
    ]
    
    missing_files = [f for f in required_files 
                     if not os.path.exists(os.path.join(output_dir, f))]
    
    if missing_files:
        raise FileNotFoundError(
            f"❌ Fichiers manquants : {missing_files}\n"
            "Exécutez d'abord : python main.py && "
            "python -c 'from src.preprocessing import prepare_data; prepare_data(...)'"
        )
    
    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    y_train = np.load(os.path.join(output_dir, "y_train.npy"))
    X_val = np.load(os.path.join(output_dir, "X_val.npy"))
    y_val = np.load(os.path.join(output_dir, "y_val.npy"))
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))
    class_weight = np.load(os.path.join(output_dir, "class_weight.npy"), 
                          allow_pickle=True).item()
    
    print(f"✅ X_train : {X_train.shape}")
    print(f"✅ y_train : {y_train.shape} ({np.mean(y_train)*100:.1f}% positifs)")
    print(f"✅ X_val : {X_val.shape}")
    print(f"✅ X_test : {X_test.shape}")
    print(f"✅ Class weights : {class_weight}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, class_weight


def load_merged_dataset(
    path: str = "data/processed/merged_dataset.csv"
) -> pd.DataFrame:
    """
    Charge le dataset fusionné et enrichi
    
    Args:
        path : Chemin vers merged_dataset.csv
    
    Returns:
        DataFrame complet (tous patients, tous séjours)
    """
    print(f"\n📊 Chargement du dataset fusionné...")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {path} n'existe pas")
    
    df = pd.read_csv(path)
    print(f"✅ Dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f"   Patients : {df['subject_id'].nunique():,}")
    print(f"   Séjours : {df['stay_id'].nunique():,}")
    print(f"   Colonnes : {list(df.columns)}")
    
    return df


def validate_sequence(
    sequence: np.ndarray,
    expected_shape: Tuple[int, int] = (24, 13)
) -> bool:
    """
    Valide qu'une séquence a la bonne forme
    
    Args:
        sequence : Array à valider
        expected_shape : Forme attendue (24 timesteps, 13 features)
    
    Returns:
        True si valide, sinon lève une exception
    """
    if sequence.shape != expected_shape:
        raise ValueError(
            f"❌ Shape attendue {expected_shape}, reçu {sequence.shape}"
        )
    
    if np.any(np.isnan(sequence)) or np.any(np.isinf(sequence)):
        raise ValueError("❌ La séquence contient NaN ou Inf")
    
    return True


def get_data_summary(output_dir: str = "data/processed") -> dict:
    """
    Affiche un résumé complet des données disponibles
    
    Returns:
        Dictionnaire avec les infos clés
    """
    print("\n" + "="*60)
    print("📈 RÉSUMÉ DES DONNÉES EARLYALERT")
    print("="*60)
    
    summary = {}
    
    try:
        X_train = np.load(os.path.join(output_dir, "X_train.npy"))
        y_train = np.load(os.path.join(output_dir, "y_train.npy"))
        X_test = np.load(os.path.join(output_dir, "X_test.npy"))
        y_test = np.load(os.path.join(output_dir, "y_test.npy"))
        
        summary["sequences_train"] = X_train.shape[0]
        summary["sequences_test"] = X_test.shape[0]
        summary["positifs_train"] = np.sum(y_train == 1)
        summary["positifs_test"] = np.sum(y_test == 1)
        summary["timesteps"] = X_train.shape[1]
        summary["features"] = X_train.shape[2]
        
        print(f"📊 Séquences d'entraînement : {summary['sequences_train']:,}")
        print(f"   ├─ Positifs (détérioration) : {summary['positifs_train']:,} ({np.mean(y_train)*100:.1f}%)")
        print(f"   └─ Négatifs (stables) : {summary['sequences_train'] - summary['positifs_train']:,}")
        print(f"\n📊 Séquences de test : {summary['sequences_test']:,}")
        print(f"   ├─ Positifs : {summary['positifs_test']:,} ({np.mean(y_test)*100:.1f}%)")
        print(f"   └─ Négatifs : {summary['sequences_test'] - summary['positifs_test']:,}")
        print(f"\n📐 Forme des séquences : {summary['timesteps']} timesteps × {summary['features']} features")
        print(f"   Durée : {summary['timesteps'] * 15} min = 6 heures")
        
    except FileNotFoundError:
        print("⚠️  Données pré-traitées non trouvées")
        summary["status"] = "data_not_prepared"
    
    print("="*60 + "\n")
    
    return summary


# ═════════════════════════════════════════════════════════════════════════════
#  UTILISATION
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Afficher le résumé des données
    summary = get_data_summary()
    
    # Exemple : Charger les données test
    try:
        X_train, y_train, X_val, y_val, X_test, y_test, class_weight = \
            load_processed_data()
        
        # Valider une séquence
        validate_sequence(X_test[0])
        print("✅ Séquence valide !")
        
    except FileNotFoundError as e:
        print(f"⚠️  {e}")

