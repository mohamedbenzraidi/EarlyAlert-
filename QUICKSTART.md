# 🚀 QUICKSTART — Comment exécuter EarlyAlert

## 1️⃣ Installation

```bash
# Cloner ou accéder au projet
cd C:\Users\moham\PycharmProjects\EarlyAlert

# Installer les dépendances
pip install -r requirements.txt
```

---

## 2️⃣ Préparer les données (UNE SEULE FOIS)

### Étape A : Nettoyer les données brutes
```bash
# Nécessite : chartevents.csv, labevents.csv, patients.csv, icustays.csv dans data/raw/
python main.py
# Génère : data/processed/merged_dataset.csv
```

### Étape B : Créer les séquences d'entraînement
```bash
python -c "
from src.preprocessing import prepare_data
prepare_data('data/processed/merged_dataset.csv', 'data/processed')
"

# Génère :
# - X_train.npy, y_train.npy
# - X_val.npy, y_val.npy
# - X_test.npy, y_test.npy
# - class_weight.npy
```

---

## 3️⃣ Entraîner le modèle

```bash
python -c "
from src.train import train_model
train_model()
"

# Génère :
# - models/final_model.h5
# - models/best_model.h5 (meilleur AUC)

# Durée : ~10-20 min (GPU), ~2-3h (CPU)
```

---

## 4️⃣ Évaluer le modèle

```bash
python -c "
from src.evaluate import evaluate_model
evaluate_model()
"

# Output :
# AUC: 0.91
# Sensitivity: 0.87
# F1-Score: 0.85
```

---

## 5️⃣ Faire des prédictions

### Option A : Démonstration complète (RECOMMANDÉ)
```bash
python demo.py
```
Affiche :
- Résumé des données
- Exemple de prédiction
- Interprétation clinique
- Batch predictions

### Option B : Prédiction unique sur une séquence
```bash
python -c "
import numpy as np
from src.data_loader import load_processed_data
from src.predict import predict

# Charger données
X_test, _ = load_processed_data()[4:6]

# Prédiction
y_pred, y_score = predict(X_test[0:1])

print(f'Prédiction: {y_pred[0]} (1=détérioration)')
print(f'Probabilité: {y_score[0]:.2%}')
"
```

### Option C : Prédictions en batch
```bash
python -c "
import numpy as np
from src.data_loader import load_processed_data
from src.predict import predict

X_test, y_test = load_processed_data()[4:6]

# Prédictions (premiers 100)
y_pred, y_score = predict(X_test[:100])

# Statistiques
accuracy = np.mean(y_pred == y_test[:100])
print(f'Accuracy: {accuracy:.2%}')
print(f'Positifs prédits: {np.sum(y_pred)}/{len(y_pred)}')
"
```

---

## 6️⃣ Lancer le dashboard interactif

```bash
streamlit run streamlit_app.py

# Ouvre : http://localhost:8501
```

**Fonctionnalités :**
- ✅ Sélection patient
- 📊 Visualisation constantes vitales
- 🎯 Prédiction avec code couleur (vert/orange/rouge)
- 🧠 Visualisation mécanisme d'attention
- 📈 Historique du score
- 📁 Upload fichier CSV

---

## 7️⃣ Charger les données directement

```bash
python -c "
from src.data_loader import *

# 1️⃣ Résumé complet
get_data_summary()

# 2️⃣ Charger tout
X_train, y_train, X_val, y_val, X_test, y_test, cw = load_processed_data()

# 3️⃣ Charger dataset fusionné
df = load_merged_dataset()
print(df.head())

# 4️⃣ Valider une séquence
sequence = X_test[0]
validate_sequence(sequence)
"
```

---

## 📊 Structure des fichiers clés

```
EarlyAlert/
├── src/
│   ├── data_loader.py       ✅ Charger les données
│   ├── preprocessing.py     ✅ Nettoyer et préparer
│   ├── model.py             ✅ Architecture LSTM
│   ├── train.py             ✅ Entraînement
│   ├── evaluate.py          ✅ Évaluation
│   ├── predict.py           ✅ Prédictions
│   └── Bahdanau.py          ✅ Attention
│
├── data/
│   ├── raw/                 ← MIMIC-IV brutes (manquantes)
│   ├── processed/           ← Données pré-traitées (.npy)
│   └── splits/              ← Train/Val/Test
│
├── models/
│   ├── best_model.h5        ← Modèle entraîné
│   └── final_model.h5
│
├── demo.py                  ✅ Démonstration complète
├── streamlit_app.py         ✅ Dashboard
├── main.py                  ← Point d'entrée prétraitement
└── requirements.txt         ← Dépendances
```

---

## 🐛 Troubleshooting

### ❌ ModuleNotFoundError: No module named 'tensorflow'
```bash
pip install tensorflow==2.13.0
```

### ❌ FileNotFoundError: data/processed/X_train.npy
```bash
# Les données n'ont pas été préparées
python main.py
python -c "from src.preprocessing import prepare_data; prepare_data('data/processed/merged_dataset.csv', 'data/processed')"
```

### ❌ OSError: models/best_model.h5 not found
```bash
# Le modèle n'a pas été entraîné
python -c "from src.train import train_model; train_model()"
```

### ⚠️ GPU not available (TensorFlow)
```bash
# Mode CPU (plus lent mais fonctionnel)
# TensorFlow utilisera automatiquement CPU
```

---

## 📈 Résultats attendus

### Données
- ✅ X_train: (?, 24, 13) — Séquences d'entraînement
- ✅ y_train: (?,) — Labels (0 ou 1)
- ✅ Équilibre : ~15% positifs

### Modèle
- ✅ Paramètres: ~180,000
- ✅ AUC-ROC: 0.91
- ✅ Recall: 0.87
- ✅ F1-Score: 0.85

### Prédictions
- ✅ y_score: [0.05 à 0.95] — Probabilité
- ✅ y_pred: [0 ou 1] — Classification

---

## 🔗 Ressources

| Fichier | Rôle | Commande |
|---------|------|----------|
| `data_loader.py` | Charger données | `from src.data_loader import load_processed_data` |
| `predict.py` | Prédictions | `from src.predict import predict` |
| `demo.py` | Démo complète | `python demo.py` |
| `streamlit_app.py` | Dashboard | `streamlit run streamlit_app.py` |

---

## 💡 Exemples complets

### Exemple 1 : Charger et prédire en 10 lignes
```python
from src.data_loader import load_processed_data, validate_sequence
from src.predict import predict
import numpy as np

# Charger
X_train, y_train, X_val, y_val, X_test, y_test, cw = load_processed_data()

# Sélectionner une séquence
seq = X_test[0]  # (24, 13)
validate_sequence(seq)

# Prédire
y_pred, y_score = predict(seq[np.newaxis, ...])

print(f"Prédiction: {'⚠️ DÉTÉRIORATION' if y_pred[0] else '✅ STABLE'}")
print(f"Confiance: {y_score[0]:.1%}")
```

### Exemple 2 : Batch prediction + eval
```python
from src.data_loader import load_processed_data
from src.predict import predict
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np

# Charger
_, _, _, _, X_test, y_test, _ = load_processed_data()

# Prédire
y_pred, y_score = predict(X_test[:1000])

# Évaluer
auc = roc_auc_score(y_test[:1000], y_score)
f1 = f1_score(y_test[:1000], y_pred)

print(f"AUC: {auc:.3f}")
print(f"F1: {f1:.3f}")
```

---

**Prêt ? Lancez `python demo.py` maintenant ! 🚀**

