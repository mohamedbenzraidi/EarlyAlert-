# 💻 Commandes PowerShell — EarlyAlert

## Navigation

```powershell
# Se déplacer dans le projet
cd C:\Users\moham\PycharmProjects\EarlyAlert

# Visualiser la structure
Get-ChildItem -Recurse -Force | Select-Object FullName | Head -30
```

---

## Installation

```powershell
# Installer toutes les dépendances
pip install -r requirements.txt

# Ou installer individuellement
pip install tensorflow==2.13.0
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install streamlit==1.25.0
```

---

## Vérifier l'installation

```powershell
# Python version
python --version

# Vérifier tensorflow
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Vérifier pandas
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"

# Vérifier GPU (si CUDA installé)
python -c "import tensorflow as tf; print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')"
```

---

## Préparer les données

```powershell
# Nettoyer les données brutes
# (Nécessite : data/raw/chartevents.csv, labevents.csv, patients.csv, icustays.csv)
python main.py

# Créer les séquences d'entraînement
python -c "
from src.preprocessing import prepare_data
prepare_data('data/processed/merged_dataset.csv', 'data/processed')
"

# Vérifier la création
Get-ChildItem data/processed/*.npy
```

---

## Entraîner le modèle

```powershell
# Entraîner simple
python -c "
from src.train import train_model
train_model()
"

# Entraîner avec temps
$start = Get-Date
python -c "from src.train import train_model; train_model()"
$end = Get-Date
Write-Host "Durée: $($end - $start)"

# Vérifier que le modèle est sauvegardé
Get-ChildItem models/*.h5
```

---

## Évaluer le modèle

```powershell
# Évaluer
python -c "
from src.evaluate import evaluate_model
evaluate_model()
"

# Afficher AUC et F1
python -c "
import numpy as np
from tensorflow.keras.models import load_model
from src.data_loader import load_processed_data
from sklearn.metrics import roc_auc_score, f1_score

model = load_model('models/best_model.h5')
_, _, _, _, X_test, y_test, _ = load_processed_data()
y_score = model.predict(X_test[:1000]).ravel()
auc = roc_auc_score(y_test[:1000], y_score)
print(f'AUC: {auc:.3f}')
"
```

---

## Faire des prédictions

### Résumé des données
```powershell
python -c "
from src.data_loader import get_data_summary
get_data_summary()
"
```

### Prédiction unique
```powershell
python -c "
import numpy as np
from src.data_loader import load_processed_data, validate_sequence
from src.predict import predict

X_test, _ = load_processed_data()[4:6]
seq = X_test[0]
validate_sequence(seq)
y_pred, y_score = predict(seq[np.newaxis, ...])
print(f'Prédiction: {y_pred[0]}')
print(f'Score: {y_score[0]:.2%}')
"
```

### Batch predictions
```powershell
python -c "
import numpy as np
from src.data_loader import load_processed_data
from src.predict import predict

X_test, y_test = load_processed_data()[4:6]
y_pred, y_score = predict(X_test[:100])
accuracy = np.mean(y_pred == y_test[:100])
print(f'Accuracy: {accuracy:.2%}')
print(f'Positifs: {np.sum(y_pred)}/{len(y_pred)}')
"
```

### Démo complète
```powershell
python demo.py
```

---

## Démonstration et Exécution

### Tous les exemples
```powershell
python RUN_EXAMPLES.py
```

### Dashboard Streamlit
```powershell
streamlit run streamlit_app.py

# Ouvre automatiquement http://localhost:8501
```

---

## Charger les données dans Python

```powershell
# Session Python interactive
python

# Puis taper :
from src.data_loader import load_processed_data, get_data_summary
summary = get_data_summary()
X_train, y_train, X_val, y_val, X_test, y_test, cw = load_processed_data()
print(X_train.shape)
exit()
```

---

## Notebook Jupyter

```powershell
# Lancer Jupyter
jupyter notebook

# Ou Jupyter Lab
jupyter lab

# Créer un notebook avec :
from src.data_loader import load_processed_data
from src.predict import predict
import numpy as np
...
```

---

## Scripts Python complets

### Script 1 : Load + Predict
```powershell
# Créer un fichier `test_predict.py`
python -c "
# test_predict.py
import numpy as np
from src.data_loader import load_processed_data
from src.predict import predict

X_test, y_test = load_processed_data()[4:6]
for i in range(5):
    pred, score = predict(X_test[i:i+1])
    print(f'{i}: True={y_test[i]}, Pred={pred[0]}, Score={score[0]:.2%}')
" > test_predict.py

# Exécuter
python test_predict.py
```

### Script 2 : Évaluation complète
```powershell
python -c "
import numpy as np
from src.data_loader import load_processed_data
from src.predict import predict
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

X_test, y_test = load_processed_data()[4:6]
y_pred, y_score = predict(X_test)

auc = roc_auc_score(y_test, y_score)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'AUC: {auc:.3f}')
print(f'F1: {f1:.3f}')
print(f'TP: {cm[1,1]}, FP: {cm[0,1]}, FN: {cm[1,0]}, TN: {cm[0,0]}')
"
```

---

## Utilitaires

### Compter les fichiers
```powershell
# Fichiers .npy
Get-ChildItem data/processed/*.npy | Measure-Object

# Fichiers .csv
Get-ChildItem data/**/*.csv | Measure-Object
```

### Taille des fichiers
```powershell
# Taille du modèle
(Get-Item models/best_model.h5).Length / 1MB

# Taille des données train
(Get-Item data/processed/X_train.npy).Length / 1MB
```

### Afficher les logs
```powershell
# Logs stderr
python RUN_EXAMPLES.py 2>&1 | Tee-Object -FilePath log.txt

# Filtre seulement erreurs
python RUN_EXAMPLES.py 2>&1 | Select-String "ERROR|Traceback"
```

---

## Nettoyage

```powershell
# Supprimer fichiers temporaires
Remove-Item -Path *.pyc -Recurse
Remove-Item -Path __pycache__ -Recurse -Force
Remove-Item -Path .pytest_cache -Recurse -Force

# Supprimer modèles (careful!)
# Remove-Item models/*.h5 -Force

# Réinitialiser les données
Remove-Item data/processed/*.npy -Force
```

---

## Troubleshooting PowerShell

### Problème : "python" not recognized
```powershell
# Solution 1 : ajouter Python au PATH manuellement
$env:PATH += ";C:\Python310\Scripts"

# Solution 2 : utiliser python directement depuis Python Launcher
py -c "import sys; print(sys.version)"
```

### Problème : Permission denied
```powershell
# Exécuter PowerShell en admin
# Puis relancer la commande
```

### Problème : pip install timeout
```powershell
# Augmenter le timeout
pip install --default-timeout=1000 tensorflow==2.13.0
```

---

## Automatiser

### Batch script (RUN_ALL.ps1)
```powershell
# Créer RUN_ALL.ps1
param(
    [ValidateSet('setup', 'train', 'eval', 'predict', 'demo')]
    [string]$step = 'demo'
)

if ($step -eq 'setup') {
    pip install -r requirements.txt
}
elseif ($step -eq 'train') {
    python -c "from src.train import train_model; train_model()"
}
elseif ($step -eq 'eval') {
    python -c "from src.evaluate import evaluate_model; evaluate_model()"
}
elseif ($step -eq 'predict') {
    python demo.py
}
elseif ($step -eq 'demo') {
    python RUN_EXAMPLES.py
}

# Utiliser :
# ./RUN_ALL.ps1 -step train
# ./RUN_ALL.ps1 -step eval
# ./RUN_ALL.ps1 -step predict
```

---

## Surveillance (Monitoring)

```powershell
# Vérifier GPU usage (pendant l'entraînement)
# NVIDIA GPUs seulement :
nvidia-smi

# Vérifier CPU/RAM
Get-Process python | Format-Table Name, CPU, Memory
```

---

## Version finale : Script tout-en-un

```powershell
# FULL_PIPELINE.ps1
Write-Host "🚀 EarlyAlert — Pipeline complet"

# 1. Setup
Write-Host "`n1️⃣ Installation des dépendances..."
pip install -r requirements.txt

# 2. Data prep (if needed)
if (-not (Test-Path "data/processed/X_train.npy")) {
    Write-Host "`n2️⃣ Préparation des données..."
    python main.py
    python -c "from src.preprocessing import prepare_data; prepare_data('data/processed/merged_dataset.csv', 'data/processed')"
} else {
    Write-Host "`n2️⃣ Données déjà préparées ✅"
}

# 3. Train
if (-not (Test-Path "models/best_model.h5")) {
    Write-Host "`n3️⃣ Entraînement du modèle..."
    python -c "from src.train import train_model; train_model()"
} else {
    Write-Host "`n3️⃣ Modèle déjà entraîné ✅"
}

# 4. Eval
Write-Host "`n4️⃣ Évaluation..."
python -c "from src.evaluate import evaluate_model; evaluate_model()"

# 5. Demo
Write-Host "`n5️⃣ Démonstration..."
python RUN_EXAMPLES.py

Write-Host "`n✅ Pipeline complet ! 🎉"
```

**Utiliser :**
```powershell
./FULL_PIPELINE.ps1
```

---

## Choisissez votre approche 🎯

| Cas d'usage | Commande |
|-------------|----------|
| **Démarrage rapide** | `python demo.py` |
| **Tous les exemples** | `python RUN_EXAMPLES.py` |
| **Dashboard interactif** | `streamlit run streamlit_app.py` |
| **Une prédiction** | `python -c "..."` (voir exemples) |
| **Workflow complet** | `./FULL_PIPELINE.ps1` |
| **Notebook** | `jupyter notebook` |

---

**Bon travail ! 🚀**

