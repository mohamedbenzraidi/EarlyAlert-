"""
DEMO — Comment utiliser data_loader.py et predict.py
Exemples concrets d'exécution du système EarlyAlert
"""

import numpy as np
import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importer les modules
from src.data_loader import (
    load_processed_data,
    load_merged_dataset,
    validate_sequence,
    get_data_summary
)
from src.predict import predict


def demo_1_load_data():
    """
    DÉMO 1 : Charger les données pré-traitées
    """
    print("\n" + "="*70)
    print("🔹 DÉMO 1 — CHARGER LES DONNÉES")
    print("="*70)
    
    try:
        # Charger TOUTES les données (train/val/test)
        X_train, y_train, X_val, y_val, X_test, y_test, class_weight = \
            load_processed_data(output_dir="data/processed")
        
        print("\n✅ Données chargées avec succès !")
        print(f"""
        📊 STATISTIQUES :
        ├─ X_train  : {X_train.shape}  → {np.sum(y_train)}  positifs ({np.mean(y_train)*100:.1f}%)
        ├─ X_val    : {X_val.shape}
        ├─ X_test   : {X_test.shape}   → {np.sum(y_test)}  positifs ({np.mean(y_test)*100:.1f}%)
        └─ Class weights : {class_weight}
        """)
        
        return X_train, y_train, X_val, y_val, X_test, y_test, class_weight
        
    except FileNotFoundError as e:
        print(f"\n❌ Erreur : {e}")
        print("\n📌 Solution : Vous devez d'abord préparer les données :")
        print("   1. python main.py")
        print("   2. python -c \"from src.preprocessing import prepare_data; prepare_data('data/processed/merged_dataset.csv', 'data/processed')\"")
        return None


def demo_2_get_one_sequence(X_test):
    """
    DÉMO 2 : Extraire une séquence test unique
    """
    print("\n" + "="*70)
    print("🔹 DÉMO 2 — EXTRAIRE UNE SÉQUENCE")
    print("="*70)
    
    # Prendre la première séquence du test
    sequence = X_test[0]  # Shape : (24, 13)
    
    print(f"""
    ✅ Séquence extraite :
    ├─ Shape : {sequence.shape} (24 timesteps × 13 features)
    ├─ Min : {sequence.min():.2f}
    ├─ Max : {sequence.max():.2f}
    └─ Mean : {sequence.mean():.2f}
    
    📝 Interprétation :
    ├─ 24 timesteps = 6 heures (15 min d'intervalle)
    ├─ 13 features = HR, SBP, SpO2, etc.
    └─ Valeurs normalisées (StandardScaler)
    """)
    
    # Valider la séquence
    try:
        validate_sequence(sequence)
        print("✅ Séquence valide !")
    except ValueError as e:
        print(f"❌ Erreur de validation : {e}")
    
    return sequence


def demo_3_predict_single(sequence):
    """
    DÉMO 3 : Faire une prédiction sur une séquence
    """
    print("\n" + "="*70)
    print("🔹 DÉMO 3 — FAIRE UNE PRÉDICTION")
    print("="*70)
    
    # Ajouter dimension batch (prédit.py attend un batch)
    X_batch = sequence[np.newaxis, ...]  # Shape : (1, 24, 13)
    
    try:
        # Prédiction
        y_pred, y_score = predict(X_batch)
        
        print(f"""
        ✅ Prédiction effectuée :
        ├─ y_pred  : {y_pred[0]} (0 = STABLE, 1 = DÉTÉRIORATION)
        ├─ y_score : {y_score[0]:.4f} (probabilité brute)
        └─ Confiance : {y_score[0]*100:.1f}%
        
        📋 Interprétation clinique :
        """)
        
        if y_score[0] < 0.35:
            print("        PATIENT STABLE — Paramétrés normaux, aucune alerte")
        elif y_score[0] < 0.65:
            print("          SURVEILLANCE RENFORCÉE — Tendance à surveiller")
        else:
            print("         ALERTE CRITIQUE — Risque de détérioration probable !")
        
        print(f"""
        🔧 Paramètres clés :
        ├─ Seuil de décision : 0.5
        └─ Score ≥ 0.5 → Prédiction positive (détérioration)
        """)
        
        return y_pred[0], y_score[0]
        
    except Exception as e:
        print(f"❌ Erreur prédiction : {e}")
        return None, None


def demo_4_predict_batch(X_test, y_test):
    """
    DÉMO 4 : Prédictions en batch sur tout l'ensemble test
    """
    print("\n" + "="*70)
    print("🔹 DÉMO 4 — PRÉDICTIONS EN BATCH")
    print("="*70)
    
    try:
        # Prédictions
        y_pred, y_score = predict(X_test[:100])  # Premiers 100 patients
        
        print(f"""
        ✅ Prédictions batch effectuées :
        ├─ Nombre de séquences : {len(y_pred)}
        ├─ Positifs prédits : {np.sum(y_pred)} ({np.mean(y_pred)*100:.1f}%)
        └─ Score moyen : {np.mean(y_score):.4f}
        """)
        
        # Comparaison vs ground truth
        y_true_batch = y_test[:100]
        accuracy = np.mean(y_pred == y_true_batch)
        
        print(f"""
        📊 Comparaison vs réalité :
        ├─ Vrai positifs : {np.sum((y_pred == 1) & (y_true_batch == 1))}
        ├─ Vrais négatifs : {np.sum((y_pred == 0) & (y_true_batch == 0))}
        ├─ Faux positifs : {np.sum((y_pred == 1) & (y_true_batch == 0))}
        ├─ Faux négatifs : {np.sum((y_pred == 0) & (y_true_batch == 1))}
        └─ Accuracy : {accuracy*100:.1f}%
        """)
        
        return y_pred, y_score
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return None, None


def demo_5_interpret_results(sequence, y_pred, y_score):
    """
    DÉMO 5 : Interpréter les résultats
    """
    print("\n" + "="*70)
    print("🔹 DÉMO 5 — INTERPRÉTATION CLINIQUE")
    print("="*70)
    
    features = ["HR (bpm)", "SBP (mmHg)", "DBP (mmHg)", "SpO2 (%)", 
                "Temp (°C)", "Resp (/min)", "Lactate", "WBC", 
                "Creatinine", "shock_index", "HR_trend", "SpO2_trend", "HR_var"]
    
    print(f"""
    🔬 RÉSULTATS DÉTAILLÉS :
    
    1️⃣  PRÉDICTION GLOBALE
    ├─ Score de risque : {y_score:.1%}
    ├─ Prédiction : {"⚠️  DÉTÉRIORATION" if y_pred == 1 else "✅ STABLE"}
    └─ Confiance : {"ÉLEVÉE" if y_score > 0.8 else "MOYENNE" if y_score > 0.5 else "FAIBLE"}
    
    2️⃣  DERNIÈRE VALEUR (T=0) — Constantes actuelles
    """)
    
    for i, feat in enumerate(features[:6]):  # Afficher les 6 premiers
        val = sequence[-1, i]  # Dernière valeur
        print(f"    {feat:20s} : {val:7.2f}")
    
    print(f"""
    3️⃣  TENDANCES — Évolution sur les 6 dernières heures
    ├─ HR tendance  : {np.mean(np.diff(sequence[:, 0])):.2f} bpm/step
    ├─ SpO2 tendance: {np.mean(np.diff(sequence[:, 3])):.2f} %/step
    └─ Temp tendance: {np.mean(np.diff(sequence[:, 4])):.2f} °C/step
    
    4️⃣  RECOMMANDATIONS CLINIQUES
    """)
    
    if y_score > 0.65:
        print("""
        🚨 ALERTE CRITIQUE
        ├─ ⚠️  Risque élevé de détérioration
        ├─ 📞 Alerter médecin/infirmier immédiatement
        ├─ 🔍 Réévaluation patient dans 15-30 min
        └─ 📊 Vérifier : FC, PA, SpO2, lactate
        """)
    elif y_score > 0.35:
        print("""
        ⚠️  SURVEILLANCE RENFORCÉE
        ├─ 📊 Danger modéré détecté
        ├─ ⏱️  Réévaluer dans 30 minutes
        ├─ 👁️  Augmenter fréquence monitoring
        └─ 💪 Considérer intervention préventive
        """)
    else:
        print("""
        ✅ PATIENT STABLE
        ├─ 😊 Aucune alerte détectée
        ├─ 📅 Monitoring standard
        ├─ 💊 Continuer traitement
        └─ 🔄 Réévaluation plan de soins
        """)


def main():
    """
    Exécuter toutes les démos
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + " DEMO — EarlyAlert Data Loader & Predict ".center(68) + "║")
    print("║" + " Système de détection précoce de détérioration en réanimation ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝\n")
    
    # Résumé des données
    get_data_summary()
    
    # DÉMO 1 : Charger les données
    result = demo_1_load_data()
    if result is None:
        print("\n❌ Impossible de continuer sans données")
        return
    
    X_train, y_train, X_val, y_val, X_test, y_test, class_weight = result
    
    # DÉMO 2 : Extraire une séquence
    sequence = demo_2_get_one_sequence(X_test)
    
    # DÉMO 3 : Prédiction simple
    y_pred, y_score = demo_3_predict_single(sequence)
    
    if y_pred is not None:
        # DÉMO 4 : Batch
        demo_4_predict_batch(X_test, y_test)
        
        # DÉMO 5 : Interprétation
        demo_5_interpret_results(sequence, y_pred, y_score)
    
    print("\n" + "="*70)
    print("✅ DÉMONSTRATION COMPLÈTE TERMINÉE")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Mode terminal
    main()
    
    # COMMANDS RAPIDES
    print("📌 COMMANDES UTILES :\n")
    print("   # Charger et résumer les données")
    print("   python -c \"from src.data_loader import get_data_summary; get_data_summary()\"\n")
    
    print("   # Faire une prédiction unique")
    print("   python -c \"")
    print("   import numpy as np")
    print("   from src.data_loader import load_processed_data")
    print("   from src.predict import predict")
    print("   X_test, y_test = load_processed_data()[4:6]")
    print("   pred, score = predict(X_test[0:1])")
    print("   print(f'Score: {score[0]:.2%}')")
    print("   \"\n")
    
    print("   # Lancer le dashboard Streamlit")
    print("   streamlit run streamlit_app.py\n")

