"""
RUN_EXAMPLES.py — Exemples d'utilisation de data_loader et predict
Copiez-collez et exécutez directement !
"""

# ==============================================================================
# EXEMPLE 1 : Charger les données et afficher résumé
# ==============================================================================

def exemple_1_charger_donnees():
    """Charger et afficher les données"""
    from src.data_loader import load_processed_data, get_data_summary
    
    print("\n" + "="*70)
    print("EXEMPLE 1 : Charger les données")
    print("="*70)
    
    # Afficher résumé
    summary = get_data_summary()
    
    # Charger tout
    X_train, y_train, X_val, y_val, X_test, y_test, class_weight = \
        load_processed_data()
    
    print(f"\n✅ Données chargées !")
    print(f"X_train: {X_train.shape} → {sum(y_train)} positifs")
    print(f"X_test: {X_test.shape} → {sum(y_test)} positifs")


# ==============================================================================
# EXEMPLE 2 : Prédire sur une séquence unique
# ==============================================================================

def exemple_2_prediction_unique():
    """Prédiction sur 1 patient"""
    import numpy as np
    from src.data_loader import load_processed_data, validate_sequence
    from src.predict import predict
    
    print("\n" + "="*70)
    print("EXEMPLE 2 : Prédiction unique")
    print("="*70)
    
    # Charger
    _, _, _, _, X_test, y_test, _ = load_processed_data()
    
    # Prendre le 1er patient test
    seq = X_test[0]  # Shape: (24, 13)
    true_label = y_test[0]
    
    # Valider
    validate_sequence(seq)
    print(f"✅ Séquence valide : {seq.shape}")
    
    # Ajouter dimension batch (predict attend (N, 24, 13))
    seq_batch = seq[np.newaxis, ...]
    
    # Prédire
    y_pred, y_score = predict(seq_batch)
    
    print(f"\n📊 Résultats :")
    print(f"  Vraie valeur : {true_label} (1=détérioration, 0=stable)")
    print(f"  Prédiction : {y_pred[0]}")
    print(f"  Score brut : {y_score[0]:.4f}")
    print(f"  Confiance : {y_score[0]*100:.1f}%")
    
    # Interprétation
    print(f"\n💡 Interprétation :")
    if y_score[0] >= 0.65:
        print(f"  🚨 ALERTE CRITIQUE — Risque élevé de détérioration")
    elif y_score[0] >= 0.35:
        print(f"  ⚠️  SURVEILLANCE RENFORCÉE — Risque modéré")
    else:
        print(f"  ✅ PATIENT STABLE — Aucune alerte")


# ==============================================================================
# EXEMPLE 3 : Batch prediction + évaluation
# ==============================================================================

def exemple_3_batch_prediction():
    """Prédictions sur plusieurs patients"""
    import numpy as np
    from src.data_loader import load_processed_data
    from src.predict import predict
    from sklearn.metrics import (
        roc_auc_score, f1_score, confusion_matrix, 
        sensitivity_specificity_support
    )
    
    print("\n" + "="*70)
    print("EXEMPLE 3 : Batch prediction (100 patients)")
    print("="*70)
    
    # Charger
    _, _, _, _, X_test, y_test, _ = load_processed_data()
    
    # Prendre 100 seqs
    X_batch = X_test[:100]
    y_truth = y_test[:100]
    
    print(f"Dataset: {X_batch.shape[0]} patients")
    
    # Prédire
    y_pred, y_score = predict(X_batch)
    
    # Évaluer
    auc = roc_auc_score(y_truth, y_score)
    f1 = f1_score(y_truth, y_pred)
    accuracy = np.mean(y_pred == y_truth)
    
    tn, fp, fn, tp = confusion_matrix(y_truth, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    print(f"\n📊 Métriques :")
    print(f"  AUC-ROC : {auc:.3f}")
    print(f"  F1-Score : {f1:.3f}")
    print(f"  Accuracy : {accuracy:.3f}")
    print(f"  Sensitivity : {sensitivity:.3f}")
    print(f"  Specificity : {specificity:.3f}")
    
    print(f"\n📈 Confusion Matrix :")
    print(f"  Vrais Négatifs  : {tn}")
    print(f"  Vrais Positifs  : {tp}")
    print(f"  Faux Positifs   : {fp}")
    print(f"  Faux Négatifs   : {fn}")
    
    print(f"\n💾 Distribution :")
    print(f"  Prédits positifs : {np.sum(y_pred)} ({np.mean(y_pred)*100:.1f}%)")
    print(f"  Prédits négatifs : {np.sum(y_pred == 0)} ({np.mean(y_pred == 0)*100:.1f}%)")


# ==============================================================================
# EXEMPLE 4 : Analyser les scores
# ==============================================================================

def exemple_4_analyser_scores():
    """Analyser la distribution des scores"""
    import numpy as np
    from src.data_loader import load_processed_data
    from src.predict import predict
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("EXEMPLE 4 : Analyser les scores")
    print("="*70)
    
    # Charger
    _, _, _, _, X_test, y_test, _ = load_processed_data()
    
    # Prédire sur 500 patients
    y_pred, y_score = predict(X_test[:500])
    y_truth = y_test[:500]
    
    # Statistiques
    print(f"\n📊 Scores des PATIENTS STABLES (y_true=0) :")
    scores_neg = y_score[y_truth == 0]
    print(f"  Min: {scores_neg.min():.3f}")
    print(f"  Max: {scores_neg.max():.3f}")
    print(f"  Mean: {scores_neg.mean():.3f}")
    print(f"  Std: {scores_neg.std():.3f}")
    
    print(f"\n📊 Scores des PATIENTS EN DÉTÉRIORATION (y_true=1) :")
    scores_pos = y_score[y_truth == 1]
    print(f"  Min: {scores_pos.min():.3f}")
    print(f"  Max: {scores_pos.max():.3f}")
    print(f"  Mean: {scores_pos.mean():.3f}")
    print(f"  Std: {scores_pos.std():.3f}")
    
    print(f"\n💡 Interprétation :")
    print(f"  Les scores des patients en détérioration")
    print(f"  sont généralement plus élevés (≥ 0.5)")
    print(f"  que ceux des patients stables (< 0.5)")
    
    # Histogramme simple
    print(f"\n📈 Distribution des scores (texte) :")
    bins = [0, 0.25, 0.5, 0.75, 1.0]
    for i in range(len(bins)-1):
        mask = (y_score >= bins[i]) & (y_score < bins[i+1])
        count = np.sum(mask)
        bar = "█" * (count // 5)
        print(f"  [{bins[i]:.2f}-{bins[i+1]:.2f}] : {bar} ({count})")


# ==============================================================================
# EXEMPLE 5 : Charger dataset fusionné (pandas)
# ==============================================================================

def exemple_5_dataset_fusionné():
    """Charger le dataset fusionné complet"""
    from src.data_loader import load_merged_dataset
    import pandas as pd
    
    print("\n" + "="*70)
    print("EXEMPLE 5 : Dataset fusionné (pandas)")
    print("="*70)
    
    # Charger
    df = load_merged_dataset()
    
    print(f"\n✅ Dataset chargé :")
    print(f"  Shape : {df.shape}")
    print(f"  Colonnes : {list(df.columns)}")
    
    print(f"\n👥 Patients :")
    print(f"  Uniques : {df['subject_id'].nunique()}")
    print(f"  Séjours : {df['stay_id'].nunique()}")
    
    print(f"\n📊 Visuel :")
    print(df.head(10))
    
    print(f"\n🔍 Stats :")
    print(df[['HR', 'SBP', 'SpO2', 'Temp', 'label']].describe())


# ==============================================================================
# EXEMPLE 6 : Validation de séquences
# ==============================================================================

def exemple_6_validation():
    """Valider des séquences"""
    import numpy as np
    from src.data_loader import validate_sequence, load_processed_data
    
    print("\n" + "="*70)
    print("EXEMPLE 6 : Validation de séquences")
    print("="*70)
    
    _, _, _, _, X_test, _, _ = load_processed_data()
    
    # OK
    print("\n✅ Test 1 : Séquence valide")
    try:
        seq = X_test[0]
        validate_sequence(seq)
        print(f"  Shape: {seq.shape} → ✅ VALIDE")
    except ValueError as e:
        print(f"  ❌ Erreur: {e}")
    
    # Pas assez de features
    print("\n❌ Test 2 : Pas assez de features")
    try:
        seq = X_test[0][:, :10]  # Enlever 3 features
        validate_sequence(seq)
    except ValueError as e:
        print(f"  ❌ Erreur: {e}")
    
    # NaN
    print("\n❌ Test 3 : Avec NaN")
    try:
        seq = X_test[0].copy()
        seq[0, 0] = np.nan
        validate_sequence(seq)
    except ValueError as e:
        print(f"  ❌ Erreur: {e}")
    
    # Inf
    print("\n❌ Test 4 : Avec Infini")
    try:
        seq = X_test[0].copy()
        seq[0, 0] = np.inf
        validate_sequence(seq)
    except ValueError as e:
        print(f"  ❌ Erreur: {e}")


# ==============================================================================
# EXEMPLE 7 : Workflow complet
# ==============================================================================

def exemple_7_workflow_complet():
    """Workflow complet : data → validation → prédiction → interprétation"""
    import numpy as np
    from src.data_loader import (
        load_processed_data, validate_sequence, get_data_summary
    )
    from src.predict import predict
    
    print("\n" + "="*70)
    print("EXEMPLE 7 : Workflow complet")
    print("="*70)
    
    print("\n1️⃣  RÉSUMÉ DES DONNÉES")
    get_data_summary()
    
    print("\n2️⃣  CHARGEMENT")
    X_train, y_train, _, _, X_test, y_test, cw = load_processed_data()
    print(f"  ✅ X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    print("\n3️⃣  SÉLECTION")
    # Prendre 3 patients : 1 stable, 1 détérioration, 1 mixte
    idx_stable = np.where(y_test == 0)[0][0]
    idx_deter = np.where(y_test == 1)[0][0]
    
    seq_stable = X_test[idx_stable]
    seq_deter = X_test[idx_deter]
    
    print(f"  Patient stable (idx={idx_stable}): {seq_stable.shape}")
    print(f"  Patient détérioration (idx={idx_deter}): {seq_deter.shape}")
    
    print("\n4️⃣  VALIDATION")
    validate_sequence(seq_stable)
    validate_sequence(seq_deter)
    print(f"  ✅ Toutes les séquences valides")
    
    print("\n5️⃣  PRÉDICTIONS")
    pred_s, score_s = predict(seq_stable[np.newaxis, ...])
    pred_d, score_d = predict(seq_deter[np.newaxis, ...])
    
    print(f"  Patient stable :")
    print(f"    Vraie valeur : 0 (stable)")
    print(f"    Prédiction : {pred_s[0]}")
    print(f"    Score : {score_s[0]:.1%}")
    
    print(f"\n  Patient détérioration :")
    print(f"    Vraie valeur : 1 (détérioration)")
    print(f"    Prédiction : {pred_d[0]}")
    print(f"    Score : {score_d[0]:.1%}")
    
    print("\n6️⃣  INTERPRÉTATION")
    print(f"  Cas 1 (Stable) :")
    if score_s[0] < 0.35:
        print(f"    ✅ Correct ! Score faible pour patient stable")
    else:
        print(f"    ⚠️  Faux positif (prédit risque sur patient stable)")
    
    print(f"\n  Cas 2 (Détérioration) :")
    if score_d[0] > 0.65:
        print(f"    ✅ Correct ! Score élevé pour patient en détérioration")
    else:
        print(f"    ⚠️  Faux négatif (prédit stable sur patient en détérioration)")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║          🏥 EARLYALERT — EXEMPLES D'UTILISATION                   ║
    ║                                                                    ║
    ║  data_loader.py : Charger les données                             ║
    ║  predict.py : Faire des prédictions                               ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Exécuter les exemples
    try:
        exemple_1_charger_donnees()
        exemple_2_prediction_unique()
        exemple_3_batch_prediction()
        exemple_4_analyser_scores()
        exemple_5_dataset_fusionné()
        exemple_6_validation()
        exemple_7_workflow_complet()
        
        print("\n" + "="*70)
        print("✅ TOUS LES EXEMPLES ONT RÉUSSI")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()

