# EarlyAlert — Rapport global (données, modèle, métriques, optimisation)

Ce document décrit le pipeline **MIMIC → séquences 24×13 → LSTM + attention Bahdanau**, les choix d’optimisation déjà en place, et comment **interpréter** les métriques (notamment le piège du seuil τ ≈ 0,05 avec un F1 test artificiellement élevé).

---

## 1. Vue d’ensemble

- **Objectif** : prédire une **détérioration** dans un horizon lié à une fenêtre de **24 pas** (résolution **15 min** → **6 h** de signal).
- **Entrée du modèle** : tenseur `(24, 13)` par fenêtre, **standardisé** avec un `StandardScaler` ajusté **uniquement sur le train** (pas de fuite val/test).
- **Sortie** : probabilité entre 0 et 1 (sigmoïde).
- **Inférence** : le seuil binaire peut être lu depuis `data/processed/inference_threshold.json` (voir section 5).

---

## 2. Données : nettoyage, fusion, taille (~11 900 lignes)

### Est-ce que « seulement » ~11 900 lignes dans `merged_dataset` explique une AUC ~0,56 ?

**Pas directement.** Ce qui compte pour l’apprentissage supervisé, ce n’est pas tant le nombre de **lignes du CSV fusionné** que :

1. **Le nombre de fenêtres** `(24 × 13)` produites après resampling, imputation, glissement, et **filtrage par séjour** (`stay_id`).
2. **La prévalence du label** (proportion de `y = 1`) et sa **stabilité** d’un séjour à l’autre.
3. **La qualité du lien** entre signaux passés et **événement futur** défini par le label.

Un fichier fusionné **court** peut quand même générer **beaucoup de fenêtres** si chaque séjour ICU a assez de points temporels. Inversement, beaucoup de lignes **redondantes** ou **très lissées** (ffill/bfill) peuvent réduire l’information utile.

### Chaîne actuelle (résumé)

- **Nettoyage** `chartevents` / `labevents` : plages physiologiques par `itemid`.
- **Fusion** : concat temporelle, **resampling 15 min** par `subject_id`, jointure **patients** / **icustays**, filtre **charttime** dans `[intime, outtime]`.
- **Imputation** : d’abord par **`stay_id`** quand présent (évite de mélanger plusieurs séjours d’un même patient), puis médianes / zéros si besoin.
- **Features dérivées** : `shock_index`, tendances FC/SpO2, variabilité FC, etc.
- **Label** : événement de détérioration à l’instant `t` défini comme **OU logique** de critères (shock index, lactate, SpO2, fréquence respiratoire), puis **fenêtre future** (logique groupe par `stay_id`) pour savoir si une détérioration survient dans l’horizon cible.

Si l’AUC reste proche de **0,55–0,60**, la cause principale est en général **la définition du label** et/ou le **bruit clinique**, pas seulement « 11 900 lignes ».

---

## 3. Optimisations déjà implémentées (code)

| Thème | Détails |
|--------|---------|
| **NaN / Inf** | Imputation par séjour, médiane globale, pas de `StandardScaler` sur des NaN ; `shock_index` sans division par SBP nul. |
| **Fuite du scaler** | `StandardScaler.fit` sur **les seules lignes train** (toutes les cellules des fenêtres train aplanies), puis `transform` val/test. |
| **Attention Bahdanau** | Terme complet `V(tanh(W1 h_i + W2 s))` avec **s** = dernier pas de temps. |
| **Sur-apprentissage** | LSTM plus petits, dropout, L2, bruit gaussien en entrée (entraînement), `recurrent_dropout` léger. |
| **Perte** | `BinaryFocalLoss` (sérialisable) + **reshape explicite** `(N,1)` pour éviter le bug de broadcast `(N,N)`. |
| **Optimiseur** | `AdamW` si disponible, sinon `Adam` ; `clipnorm` ; `ReduceLROnPlateau` sur **`val_auc`** ; early stopping sur **`val_auc`**. |
| **GPU Windows** | TensorFlow **≥ 2.11** n’utilise **pas** CUDA sur Windows natif : message dans `train` ; solutions possibles : **WSL2** + build CUDA, ou **DirectML** selon versions. |
| **Métriques post-entraînement** | AUC test recalculée avec **sklearn** ; seuils sauvegardés dans `inference_threshold.json`. |

---

## 4. Interpréter vos derniers chiffres (exemple τ = 0,05, F1 test 0,80, balanced acc 0,50)

### Piège classique : F1 élevé + seuil très bas

- Si le modèle sort des **probabilités globalement basses** (mal calibrées ou séparabilité faible), maximiser le **F1 sur la validation** peut pousser τ vers **0,05** : on classe **presque tout** comme positif.
- Conséquences typiques :
  - **Accuracy** et **F1** peuvent **monter** sur un jeu où la classe majoritaire « colle » à cette stratégie ;
  - **`balanced_accuracy` ≈ 0,5** indique que la performance **par classe** (moyenne des rappels) reste **proche du hasard** — le seuil « optimal F1 » **n’améliore pas** réellement la discrimination globale.

**L’AUC ~0,56** reste l’indicateur le plus honnête ici : le **score** ne sépare que faiblement les deux classes, **quel que soit** le seuil.

### Correction dans le code (seuil pour `predict()`)

Le seuil sauvegardé pour l’inférence n’est plus choisi par **F1 seul sur toute la plage** (qui autorisait des τ extrêmes). Il maximise sur la validation une combinaison **0,5 × balanced_accuracy + 0,5 × F1**, avec τ **borné entre 0,15 et 0,85**. Les valeurs **Youden** et **F1-only** (dans la même plage) sont conservées en **référence** dans le JSON.

---

## 5. Fichiers générés utiles

| Fichier | Rôle |
|---------|------|
| `data/processed/X_*.npy`, `y_*.npy` | Fenêtres train/val/test. |
| `data/processed/feature_scaler.joblib` | Standardisation alignée entraînement / Streamlit. |
| `data/processed/feature_columns.json` | Ordre des 13 colonnes. |
| `data/processed/class_weight.npy` | Poids de classes (utilisés dans `fit`). |
| `data/processed/training_metrics.json` | Loss test (Keras), AUC sklearn, accuracies @0,5 et @τ, F1, balanced acc, τ. |
| `data/processed/inference_threshold.json` | `decision_threshold` pour `predict.py`, plus métadonnées (Youden, F1-only, etc.). |
| `models/best_model.h5`, `models/final_model.h5` | Modèles Keras (couche Bahdanau + perte focal enregistrée). |

---

## 6. Comment pousser la performance (ordre recommandé)

1. **Vérifier le label** : prévalence, sens clinique, horizon (6 h vs autre), critères (shock index seul vs composite).
2. **Courbes sur la validation** : ROC, PR, calibration (reliability diagram) ; ne pas se fier à un seul couple (F1, τ).
3. **Cohortes** : exclure les séjours trop courts pour 24 pas ; homogénéiser la population si besoin.
4. **Features** : ajouter des signaux validés cliniquement ; éviter des fuites temporelles (données « futures » dans la fenêtre).
5. **Modèle** : baseline **logistique** ou **XGBoost** sur le dernier pas pour situer un plafond d’AUC avant de complexifier le LSTM.

---

## 7. Relancer le pipeline (rappel)

1. Nettoyage + fusion (`main.py` ou étapes équivalentes).
2. **`prepare_data`** → régénère `.npy`, scaler, colonnes.
3. **`train_model`** → met à jour métriques, seuils, `final_model.h5`.
4. **`streamlit run streamlit_app.py`** (depuis la racine du projet) pour l’UI ; **`predict`** lit le seuil depuis `inference_threshold.json`.

---

*Document synthétique — à tenir à jour lorsque le label ou les features changent.*
