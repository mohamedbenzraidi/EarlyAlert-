# Rapport — Amélioration du modèle EarlyAlert et alignement UI / backend

## 1. Pourquoi le train affichait ~99 % d’accuracy alors que le test restait ~0,5–0,6 ?

### Sur-apprentissage (overfitting)

Les courbes du type **train accuracy ≈ 0,99**, **train AUC ≈ 1**, alors que **val_accuracy ≈ 0,49**, **val_auc ≈ 0,45** et **val_loss élevée (~3+)** indiquent que le modèle **mémorise fortement les exemples d’entraînement** sans généraliser aux séjours de validation.

- Sur le **train**, avec `class_weight`, le réseau peut séparer des motifs très fins (voire du bruit) et les métriques **accuracy / AUC** montent.
- Sur la **validation** (autres séjours `stay_id`), ces motifs ne se reproduisent pas : les probabilités sont **mal calibrées**, la **loss** (BCE non pondérée à l’évaluation) explose, et l’**AUC** retombe vers le hasard ou en dessous.

Le **jeu de test** (autre tirage de séjours) peut donner un **AUC ~0,63** : c’est une **autre estimation** de la généralisation, en général un peu plus optimiste ou différente selon la répartition des cas difficiles.

### Arrêt à l’epoch 20

L’**early stopping** surveille **`val_auc`** (avec `restore_best_weights=True`). Dès que **`val_auc` ne s’améliore plus** pendant la patience configurée, l’entraînement s’arrête et les poids **du meilleur epoch sur la validation** sont restaurés — pas nécessairement le dernier epoch affiché.

---

## 2. Changements réalisés pour améliorer le modèle et la robustesse

| Zone | Problème | Changement |
|------|-----------|------------|
| **Données** | NaN / Inf → loss ~0,693, pas d’apprentissage | Imputation par séjour + médiane, `shock_index` sans division par zéro, `±Inf` → NaN (`preprocessing.py`) |
| **Attention** | W2 inutilisé, score simpliste | Bahdanau additive : `V(tanh(W1 h_i + W2 s))` avec **s = dernier pas de temps** (`Bahdanau.py`) |
| **Entraînement** | Chemins fragiles (`getcwd`) | Racine projet via `__file__` (`train.py`) |
| **Callbacks** | Plateau / pas de ralentissement LR | `ReduceLROnPlateau`, early stopping sur **`val_auc`** |
| **Sur-apprentissage** | Train ≫ val | **Dropout** augmenté (≈0,45–0,5), **L2** sur `Dense`, **recurrent_dropout** léger sur LSTM, **`BinaryCrossentropy(label_smoothing=0.05)`** (`model.py`, `train.py`) |
| **Inférence** | Pas de scaler sauvegardé, incohérence 10 vs 13 features | Sauvegarde **`feature_scaler.joblib`** + **`feature_columns.json`**, module **`src/earlyalert_inference.py`** |
| **Suivi** | Métriques inventées dans l’UI | Écriture **`data/processed/training_metrics.json`** après `evaluate` sur le test (`train.py`) |
| **`predict.py`** | Référence à `final_model()` inexistante | Chargement explicite avec **`BahdanauAttention`** dans `custom_objects` |

### Fichiers clés modifiés ou ajoutés

- `src/preprocessing.py` — imputation, scaler persisté  
- `src/Bahdanau.py` — attention complète  
- `src/model.py` — régularisation  
- `src/train.py` — loss lissée, métriques JSON, garde NaN  
- `src/earlyalert_inference.py` — **nouveau** : chargement modèle + scaler, conversion démo 10 → 13, prédiction  
- `streamlit_app.py` — bundle modèle/scaler, métriques réelles, CSV 24×10 ou 24×13  
- `src/predict.py` — chargement cohérent avec l’évaluation  

---

## 3. Alignement Streamlit (front) et backend

### Avant

- Chargement **`load_model()`** sans **`custom_objects`** → échec probable avec une couche **`BahdanauAttention`** personnalisée.  
- Entrée **(24, 10)** alors que le réseau attend **(24, 13)** après le pipeline MIMIC + **StandardScaler**.  
- Métriques **fixes** (ex. AUC 0,91) non lues depuis l’entraînement réel.

### Après

- Chargement de **`best_model.h5`** ou **`final_model.h5`** avec **`BahdanauAttention`**.  
- Utilisation de **`feature_scaler.joblib`** lorsqu’il est présent (recommandé après `prepare_data`).  
- Séquences **24×10** (démo UI) : mapping documenté vers **13 colonnes brutes** puis **transform scaler** (approximation clinique pour la démo uniquement).  
- Import CSV **24×13** : ordre aligné sur le pipeline (**HR, SBP, DBP, SpO2, Temp, Resp, Creatinine, Lactate, WBC, shock_index, HR_trend, SpO2_trend, HR_var**).  
- Bandeau latéral / pied de page : métriques lues dans **`training_metrics.json`** si disponible.

### Attention (explainabilité dans l’UI)

Le **`Model` Keras** final n’expose pas la sortie intermédiaire des poids d’attention. L’onglet « Attention » affiche désormais un **proxy** basé sur la **variation temporelle** des signaux (et l’ancienne simulation en **mode démo** sans modèle).

---

## 4. Prochaines étapes recommandées

1. **Regénérer les données** : `prepare_data` → produit `feature_scaler.joblib`.  
2. **Ré-entraîner** et vérifier que **train et val** se rapprochent (AUC val > 0,55 typiquement souhaitable avant déploiement).  
3. Si le fossé train/val persiste : **moins de paramètres**, **plus de données**, **validation croisée par séjour**, ou révision de la **définition du label** (horizon 6 h, shock index, etc.).

---

*Document généré dans le cadre du projet EarlyAlert — cohérence modèle / pipeline / Streamlit.*
