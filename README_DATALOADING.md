# 📚 Documentation Complète — data_loader.py & predict.py

## 🏥 EarlyAlert — Vue d'ensemble

**Objectif** : Prédire une détérioration critique chez des patients en réanimation **4-6 heures avant** qu'elle ne soit cliniquement visible.

**Technologie** : LSTM Bidirectionnel + Mécanisme d'Attention Bahdanau

**Données** : MIMIC-IV (300k+ séjours réels, constantes vitales + biomarqueurs)

---

## 📦 `data_loader.py` — Chargement des données

C'est le module qui charge les données depuis les fichiers sauvegardés.

### Fonction 1️⃣ : `load_raw_data()`

**Objectif** : Charger les 4 tables brutes de MIMIC-IV

```python
from src.data_loader import load_raw_data

chartevents, labevents, patients, icustays = load_raw_data(
    path_chartevents="data/raw/chartevents.csv",
    path_labevents="data/raw/labevents.csv",
    path_patients="data/raw/patients.csv",
    path_icustays="data/raw/icustays.csv"
)
```

**Retourne** :
- `chartevents` (DataFrame) : Constantes vitales (HR, TA, SpO2, Temp, FR)
- `labevents` (DataFrame) : Résultats biologiques (Lactate, WBC, Creatinine)
- `patients` (DataFrame) : Données démographiques (âge, sexe, mortalité)
- `icustays` (DataFrame) : Séjours en réanimation (dates d'admission/sortie)

**Erreur possible** :
```
FileNotFoundError: ❌ Un ou plusieurs fichiers .csv sont manquants dans data/raw/
```
**Solution** : Télécharger les fichiers MIMIC-IV depuis PhysioNet

---

### Fonction 2️⃣ : `load_processed_data()`

**Objectif** : Charger les données **pré-traitées et préparées** pour l'entraînement

```python
from src.data_loader import load_processed_data

X_train, y_train, X_val, y_val, X_test, y_test, class_weight = load_processed_data(
    output_dir="data/processed"
)
```

**Retourne** :
| Variable | Shape | Description |
|----------|-------|-------------|
| `X_train` | (N, 24, 13) | Séquences d'entraînement : N patients × 24 timesteps × 13 features |
| `y_train` | (N,) | Labels (0 = stable, 1 = détérioration dans 6h) |
| `X_val` | (M, 24, 13) | Séquences de validation |
| `y_val` | (M,) | Labels validation |
| `X_test` | (P, 24, 13) | Séquences de test |
| `y_test` | (P,) | Labels test |
| `class_weight` | dict | Poids pour équilibre classes : `{0: 0.36, 1: 1.64}` |

**Exemple d'utilisation** :
```python
# Charger les données
X_train, y_train, X_val, y_val, X_test, y_test, cw = load_processed_data()

# Afficher infos
print(f"X_train shape: {X_train.shape}")  # (45000, 24, 13)
print(f"Positifs: {sum(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")  # ~15%
print(f"Class weights: {cw}")  # {0: 0.36, 1: 1.64}

# Accéder une séquence patient
seq = X_train[0]  # (24, 13)
label = y_train[0]  # 0 ou 1
```

---

### Fonction 3️⃣ : `load_merged_dataset()`

**Objectif** : Charger le **dataset fusionné complet** (avant split train/val/test)

```python
from src.data_loader import load_merged_dataset

df = load_merged_dataset(path="data/processed/merged_dataset.csv")
```

**Retourne** : DataFrame pandas avec colonnes :
```
Colonnes : subject_id, stay_id, charttime, HR, SBP, DBP, SpO2, Temp, Resp, 
           Lactate, WBC, Creatinine, shock_index, HR_trend, SpO2_trend, HR_var, label
```

**Utilisation** :
```python
# Identifier patients uniques
print(f"Patients: {df['subject_id'].nunique()}")  # ex: 12000

# Voir les features
print(df.columns.tolist())

# Filtrer par patient
pat = df[df['subject_id'] == 10001]
print(pat[['charttime', 'HR', 'SpO2', 'label']])
```

---

### Fonction 4️⃣ : `validate_sequence()`

**Objectif** : **Vérifier qu'une séquence est valide** avant la prédiction

```python
from src.data_loader import validate_sequence
import numpy as np

# Extraire une séquence
seq = X_test[0]  # Shape : (24, 13)

# Valider
try:
    validate_sequence(seq, expected_shape=(24, 13))
    print("✅ Séquence valide !")
except ValueError as e:
    print(f"❌ Erreur: {e}")
```

**Vérifie** :
- ✅ Shape = (24, 13)
- ✅ Pas de NaN
- ✅ Pas de valeurs infinies

**Erreurs possibles** :
```
ValueError: ❌ Shape attendue (24, 13), reçu (24, 12)  # Mauvais nombre de features
ValueError: ❌ La séquence contient NaN ou Inf  # Données corrompues
```

---

### Fonction 5️⃣ : `get_data_summary()`

**Objectif** : **Afficher un résumé complet des données**

```python
from src.data_loader import get_data_summary

summary = get_data_summary(output_dir="data/processed")
```

**Affiche** :
```
============================================================
📈 RÉSUMÉ DES DONNÉES EARLYALERT
============================================================
📊 Séquences d'entraînement : 45,000
   ├─ Positifs (détérioration) : 6,750 (15.0%)
   └─ Négatifs (stables) : 38,250 (85.0%)

📊 Séquences de test : 15,000
   ├─ Positifs : 2,250 (15.0%)
   └─ Négatifs : 12,750 (85.0%)

📐 Forme des séquences : 24 timesteps × 13 features
   Durée : 360 min = 6 heures
============================================================
```

**Retourne** : Dictionnaire Python
```python
{
    'sequences_train': 45000,
    'sequences_test': 15000,
    'positifs_train': 6750,
    'positifs_test': 2250,
    'timesteps': 24,
    'features': 13
}
```

---

## 🔬 `predict.py` — Faire des prédictions

C'est le module qui charge le modèle entraîné et fait des prédictions.

### Fonction : `predict(X)`

**Objectif** : **Prédire le risque de détérioration pour un ou plusieurs patients**

```python
from src.predict import predict
import numpy as np

# Charger une séquence (24 timesteps × 13 features)
from src.data_loader import load_processed_data
X_test, _ = load_processed_data()[4:6]

# Prédiction simple (1 patient)
y_pred, y_score = predict(X_test[0:1])
print(f"Prédiction: {y_pred[0]}")  # 0 ou 1
print(f"Score: {y_score[0]:.4f}")  # 0.000 à 1.000

# Ou batch (plusieurs patients en même temps)
y_pred, y_score = predict(X_test[:100])
print(f"Prédictions: {y_pred}")  # [0, 1, 0, 1, ...]
print(f"Scores: {y_score}")  # [0.15, 0.78, 0.22, ...]
```

**Entrée** :
- `X` (numpy array) : Shape (N, 24, 13)
  - N = nombre de patients/séquences
  - 24 = timesteps (6h)
  - 13 = features

**Sortie** :
- `y_pred` (array) : Prédictions binaires [0, 1] (0 = stable, 1 = détérioration)
- `y_score` (array) : Probabilités brutes [0.0 à 1.0]

**Exemple complet** :
```python
from src.data_loader import load_processed_data, validate_sequence
from src.predict import predict
import numpy as np

# 1. Charger les données
X_train, y_train, X_val, y_val, X_test, y_test, _ = load_processed_data()

# 2. Sélectionner UNE séquence patient
seq = X_test[5]  # Patient #5, shape (24, 13)
true_label = y_test[5]

# 3. Valider
validate_sequence(seq)

# 4. Prédire
y_pred, y_score = predict(seq[np.newaxis, ...])  # ⚠️ Ajouter dimension batch !

# 5. Interpréter
print(f"Vraie valeur: {true_label} (1=détérioration, 0=stable)")
print(f"Prédiction: {y_pred[0]}")
print(f"Confiance: {y_score[0]:.1%}")

if y_score[0] > 0.65:
    print("🚨 ALERTE CRITIQUE — Risque élevé")
elif y_score[0] > 0.35:
    print("⚠️  SURVEILLANCE — Risque modéré")
else:
    print("✅ STABLE — Aucun risque")
```

---

## 🎯 Workflow complet : Data → Predict

```
┌──────────────────────────────────────────────────────────────┐
│ 1. CHARGER LES DONNÉES                                       │
│    from src.data_loader import load_processed_data           │
│    X_train, y_train, X_test, y_test = ...                    │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│ 2. VALIDER UNE SÉQUENCE                                      │
│    from src.data_loader import validate_sequence             │
│    sequence = X_test[0]  # shape (24, 13)                    │
│    validate_sequence(sequence)                               │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│ 3. PRÉDIRE                                                   │
│    from src.predict import predict                           │
│    y_pred, y_score = predict(sequence[np.newaxis, ...])      │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│ 4. INTERPRÉTER                                               │
│    if y_score[0] > 0.65:                                     │
│        print("🚨 ALERTE CRITIQUE")                           │
│    elif y_score[0] > 0.35:                                   │
│        print("⚠️  SURVEILLANCE")                             │
│    else:                                                     │
│        print("✅ STABLE")                                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 13 Features expliquées

### Constantes Vitales (Vitals)
| # | Feature | Unité | Normal | Rôle |
|----|---------|-------|--------|------|
| 0 | HR | bpm | 60-100 | Fréquence cardiaque |
| 1 | SBP | mmHg | 90-140 | Tension artérielle systolique |
| 2 | DBP | mmHg | 60-90 | Tension artérielle diastolique |
| 3 | SpO2 | % | 95-100 | Saturation en oxygène² |
| 4 | Temp | °C | 36-37.5 | Température |
| 5 | Resp | /min | 12-20 | Fréquence respiratoire |

### Biomarqueurs (Labs)
| # | Feature | Unité | Normal | Rôle |
|----|---------|-------|--------|------|
| 6 | Lactate | mmol/L | 0.5-2 | Indicateur de choc |
| 7 | WBC | K/μL | 4-11 | Globules blancs (infection) |
| 8 | Creatinine | mg/dL | 0.6-1.2 | Fonction rénale |

### Dérivées (Calculated)
| # | Feature | Unité | Rôle |
|----|---------|-------|------|
| 9 | shock_index | unitless | FC/SBP (>1 = choc) |
| 10 | HR_trend | bpm/15min | Dérivée FC |
| 11 | SpO2_trend | %/15min | Dérivée SpO2 |
| 12 | HR_var | bpm | Variabilité FC (std 4-window) |

---

## 🔢 Les 24 Timesteps

Chaque séquence contient **24 observations** (1 par 15 minutes = 6 heures) :

```
T-360min (6h ago)  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │ T-0 (NOW)
┌──────────────────┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼││ │  │  │  │ │T
│ Timestep 0        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 (Index)
└────────────────────────────────────────────────────────────────────────────────────────┘
│
└─ Pour CHAQUE timestep : (HR, SBP, DBP, SpO2, Temp, Resp, Lactate, WBC, Creat., shock_idx, HR_tr, SpO2_tr, HR_var)
   = 13 valeurs
```

**Example** : Pour le patient 001, on prend :
- Observations du patient de T-6h à T-0
- Export = array (24, 13)
- Si détérioration → y_true = 1
- Si pas détérioration → y_true = 0

---

## 🎯 Utilisation pratique

### Script 1️⃣ : Charger et afficher
```python
from src.data_loader import load_processed_data, get_data_summary
import pandas as pd

# Résumé
summary = get_data_summary()

# Charger
X_train, y_train, _, _, X_test, y_test, cw = load_processed_data()

# Afficher infos
print(f"Total patients: {len(X_train) + len(X_test)}")
print(f"Déséquilibre: {cw}")

# Premier patient entraînement
print(f"Patient 0: {X_train[0].shape}, Label: {y_train[0]}")
```

### Script 2️⃣ : Prédiction sur plusieurs patients
```python
from src.data_loader import load_processed_data
from src.predict import predict
from sklearn.metrics import roc_auc_score, f1_score

# Charger
_, _, _, _, X_test, y_test, _ = load_processed_data()

# Prédire (premiers 1000)
y_pred, y_score = predict(X_test[:1000])

# Évaluer
auc = roc_auc_score(y_test[:1000], y_score)
f1 = f1_score(y_test[:1000], y_pred)
recall = sum((y_pred == 1) & (y_test[:1000] == 1)) / sum(y_test[:1000] == 1)

print(f"AUC: {auc:.3f}")
print(f"F1: {f1:.3f}")
print(f"Recall: {recall:.3f}")
```

### Script 3️⃣ : Utiliser dans Streamlit
```python
import streamlit as st
import numpy as np
from src.data_loader import load_processed_data, validate_sequence
from src.predict import predict

# Sidebar
st.sidebar.title("Upload Séquence")
uploaded = st.sidebar.file_uploader("CSV (24x13)", type=["csv"])

if uploaded:
    # Charger
    seq = pd.read_csv(uploaded, header=None).values.astype(float)
    
    # Valider
    validate_sequence(seq)
    
    # Prédire
    y_pred, y_score = predict(seq[np.newaxis, ...])
    
    # Afficher
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", f"{y_score[0]:.1%}")
    with col2:
        if y_score[0] > 0.65:
            st.error("🚨 CRITIQUE")
        else:
            st.success("✅ OK")
```

---

## 🐛 Erreurs courantes et solutions

| Erreur | Cause | Solution |
|--------|-------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | TensorFlow pas installé | `pip install tensorflow==2.13.0` |
| `FileNotFoundError: X_train.npy not found` | Données pas préparées | `python main.py && python -c "..."` |
| `ValueError: Shape attendue (24, 13), reçu (24, 12)` | Mauvaise shape | Vérifier le fichier CSV |
| `ValueError: La séquence contient NaN` | Données corrompues | Nettoyer le CSV |
| `OSError: models/best_model.h5 not found` | Modèle pas entraîné | `python -c "from src.train import train_model; train_model()"` |

---

## 📚 Ressources

- 📖 Readme complet : `readme.md`
- 🚀 Démarrage rapide : `QUICKSTART.md`
- 🔧 Démo complète : `demo.py`
- 🎨 Dashboard : `streamlit run streamlit_app.py`

**Bon développement ! 🚀**

