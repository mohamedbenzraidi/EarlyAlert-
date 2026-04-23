# 🏥 EarlyAlert — Détection Précoce de Détérioration en Réanimation

> Système de prédiction de détérioration patient basé sur un **LSTM Bidirectionnel + Mécanisme d'Attention**,  
> entraîné from scratch sur des données réelles de réanimation (MIMIC-IV).

---

## 🎯 Objectif du Projet

Prédire une détérioration critique d'un patient en réanimation **4 à 6 heures avant** qu'elle soit cliniquement visible,  
en analysant l'évolution temporelle de ses constantes vitales.

**Problème résolu** : Les infirmières ne peuvent pas surveiller 10 patients simultanément en permanence.  
EarlyAlert agit comme un système d'alerte précoce automatisé et explicable.

---

## 🛠️ Stack Technique Complète

| Catégorie | Outil | Rôle |
|-----------|-------|------|
| **Données** | MIMIC-IV (PhysioNet) | Base de données clinique réelle — 300k+ séjours en réanimation |
| **Langage** | Python 3.10 | Langage principal du backend/ML |
| **Deep Learning** | TensorFlow / Keras | Construction et entraînement du modèle LSTM |
| **Traitement données** | Pandas, NumPy | Manipulation et transformation des séquences temporelles |
| **ML classique** | Scikit-learn | Prétraitement, split, métriques, baselines |
| **Imputation** | KNNImputer (sklearn) | Gestion des valeurs manquantes dans les séries temporelles |
| **Visualisation** | Matplotlib, Seaborn | Courbes d'apprentissage, matrices de confusion, ROC |
| **Visualisation interactive** | Recharts / Plotly.js | Graphiques dynamiques dans le dashboard React |
| **Frontend** | React 18 + Vite | Dashboard médical interactif |
| **UI Components** | shadcn/ui + Tailwind CSS | Composants accessibles et stylisés |
| **API Backend** | FastAPI | Exposition du modèle en REST API |
| **Environnement GPU** | Kaggle Notebooks | GPU T4 gratuit pour l'entraînement |
| **Déploiement** | Hugging Face Spaces + Docker | Hébergement public de la démo |
| **Versioning** | GitHub | Gestion du code source |

---

## 📁 Structure du Projet

```
EarlyAlert/
│
├── data/
│   ├── raw/                    # Données brutes MIMIC-IV (chartevents, icustays...)
│   ├── processed/              # Données nettoyées et séquences construites
│   └── splits/                 # Train / Validation / Test
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploration et analyse des données
│   ├── 02_Preprocessing.ipynb  # Nettoyage et feature engineering
│   ├── 03_Training.ipynb       # Entraînement du modèle LSTM
│   └── 04_Evaluation.ipynb     # Métriques et visualisations
│
├── src/
│   ├── data_loader.py          # Chargement et extraction MIMIC-IV
│   ├── preprocessing.py        # Nettoyage, normalisation, séquences
│   ├── model.py                # Architecture LSTM + Attention
│   ├── train.py                # Pipeline d'entraînement
│   ├── evaluate.py             # Évaluation et métriques cliniques
│   └── predict.py              # Inférence temps réel
│
├── api/
│   ├── main.py                 # Serveur FastAPI
│   ├── schemas.py              # Modèles Pydantic (requêtes/réponses)
│   └── inference.py            # Chargement modèle + prédiction
│
├── frontend/                   # Application React
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx         # Vue principale avec grille patients
│   │   │   ├── PatientCard.tsx       # Carte patient avec score de risque
│   │   │   ├── VitalsChart.tsx       # Graphique Recharts des constantes
│   │   │   ├── AttentionHeatmap.tsx  # Visualisation des poids d'attention
│   │   │   ├── AlertBadge.tsx        # Badge vert/orange/rouge
│   │   │   └── PatientUploader.tsx   # Upload CSV ou données synthétiques
│   │   ├── hooks/
│   │   │   ├── usePatientData.ts     # Fetch données patient depuis l'API
│   │   │   └── usePrediction.ts      # Appel API prédiction en temps réel
│   │   ├── lib/
│   │   │   └── api.ts                # Client API (fetch vers FastAPI)
│   │   ├── types/
│   │   │   └── index.ts              # Types TypeScript (Patient, Vitals...)
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.ts
│
├── models/
│   └── best_model.h5           # Modèle entraîné sauvegardé
│
├── requirements.txt            # Dépendances Python
├── Dockerfile
└── README.md
```

---

## 🗓️ Roadmap — 3 Mois

### 📅 MOIS 1 — Données & Prétraitement

#### Semaine 1 — Accès aux données
- [ ] Créer un compte sur [PhysioNet.org](https://physionet.org)
- [ ] Compléter la formation CITI Program (~2h, obligatoire)
- [ ] Demander l'accès à MIMIC-IV (approuvé en 24-48h)
- [ ] Télécharger les tables nécessaires :
  - `chartevents.csv` → constantes vitales
  - `icustays.csv` → séjours en réanimation
  - `patients.csv` → données démographiques
  - `labevents.csv` → résultats biologiques

#### Semaine 2 — Exploration des données (EDA)
- [ ] Analyser la distribution des constantes vitales
- [ ] Identifier les valeurs aberrantes et manquantes
- [ ] Visualiser les patterns temporels des patients
- [ ] Définir précisément les critères de "détérioration"

#### Semaine 3 — Nettoyage & Feature Engineering
- [ ] Filtrer les valeurs aberrantes (FC > 300, SpO2 < 50...)
- [ ] Rééchantillonner les séries toutes les 15 minutes
- [ ] Appliquer KNNImputer pour les valeurs manquantes
- [ ] Créer les features dérivées : tendances, shock index, variabilité

#### Semaine 4 — Construction des séquences
- [ ] Labelliser chaque instant (détérioration dans les 6h ? oui/non)
- [ ] Construire les fenêtres glissantes de 24 timesteps (= 6h)
- [ ] Gérer le déséquilibre de classes (~15% positifs)
- [ ] Diviser en Train / Validation / Test (70/15/15)

---

### 📅 MOIS 2 — Modèle Deep Learning & API

#### Semaine 5 — Architecture LSTM
- [ ] Implémenter le modèle LSTM Bidirectionnel from scratch
- [ ] Ajouter le mécanisme d'Attention de Bahdanau
- [ ] Configurer les couches Dense + Dropout
- [ ] Vérifier le nombre de paramètres (~180k)

#### Semaine 6 — Entraînement
- [ ] Configurer les class weights pour le déséquilibre
- [ ] Entraîner sur Kaggle GPU T4 (epochs=100)
- [ ] Utiliser EarlyStopping + ReduceLROnPlateau
- [ ] Sauvegarder le meilleur modèle (ModelCheckpoint)

#### Semaine 7 — Évaluation clinique
- [ ] Calculer AUC-ROC (objectif : > 0.88)
- [ ] Calculer Sensibilité, Spécificité, F1-Score
- [ ] Tracer la courbe ROC complète
- [ ] Visualiser les poids d'attention sur des cas réels

#### Semaine 8 — API FastAPI
- [ ] Créer l'endpoint `POST /predict` (reçoit séquence → retourne score + attention)
- [ ] Créer l'endpoint `GET /patient/{id}` (données temps réel)
- [ ] Ajouter la validation des données avec Pydantic
- [ ] Tester l'API avec Swagger UI (`/docs`)

---

### 📅 MOIS 3 — Frontend React & Déploiement

#### Semaine 9-10 — Interface React
- [ ] Initialiser le projet avec Vite + React + TypeScript
- [ ] Configurer Tailwind CSS + shadcn/ui
- [ ] Dashboard principal : grille des patients avec score de risque
- [ ] Graphiques Recharts des constantes vitales (FC, SpO2, TA...)
- [ ] Composant `AlertBadge` (vert/orange/rouge selon seuil)
- [ ] Heatmap des poids d'attention par timestep
- [ ] Système d'upload CSV ou simulation de patient synthétique
- [ ] Connexion à l'API FastAPI via hooks React personnalisés

#### Semaine 11-12 — Déploiement
- [ ] Dockeriser frontend (Nginx) + backend (FastAPI) avec docker-compose
- [ ] Déployer sur Hugging Face Spaces (gratuit)
- [ ] Publier le code sur GitHub avec documentation
- [ ] Préparer la démonstration jury (3 cas patients préparés)

---

## 🧠 Architecture du Modèle

```
Input : séquence de 24 timesteps × 10 features
(= 6 heures de constantes vitales toutes les 15 min)

        ┌─────────────────────────┐
        │  Bidirectional LSTM 64  │  ← lit passé ET tendances inverses
        │  return_sequences=True  │
        └──────────┬──────────────┘
                   │
        ┌──────────▼──────────────┐
        │  LayerNormalization     │  ← stabilise l'entraînement
        │  Dropout 0.3            │
        └──────────┬──────────────┘
                   │
        ┌──────────▼──────────────┐
        │  Bidirectional LSTM 32  │  ← affine la représentation
        │  return_sequences=True  │
        └──────────┬──────────────┘
                   │
        ┌──────────▼──────────────┐
        │  Attention de Bahdanau  │  ← identifie les moments critiques
        │  Dense(1, tanh)         │
        │  Softmax → Multiply     │
        └──────────┬──────────────┘
                   │
        ┌──────────▼──────────────┐
        │  Dense 64 (relu)        │
        │  Dropout 0.3            │
        │  Dense 32 (relu)        │
        └──────────┬──────────────┘
                   │
        ┌──────────▼──────────────┐
        │  Dense 1 (sigmoid)      │  ← probabilité de détérioration
        └─────────────────────────┘

Total paramètres : ~180,000
```

---

## 🔌 Architecture Fullstack

```
┌──────────────────────────────────────────┐
│           React Frontend (Vite)          │
│  Dashboard · Graphiques · Alertes        │
│  Recharts · shadcn/ui · Tailwind CSS     │
└───────────────────┬──────────────────────┘
                    │ HTTP REST (JSON)
                    ▼
┌──────────────────────────────────────────┐
│           FastAPI Backend                │
│  POST /predict  ·  GET /patient/{id}     │
│  Pydantic · Uvicorn                      │
└───────────────────┬──────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│           Modèle TensorFlow              │
│  LSTM Bidi + Attention de Bahdanau       │
│  best_model.h5                           │
└──────────────────────────────────────────┘
```

---

## 📊 Métriques Attendues

| Modèle | AUC-ROC | Recall | F1 |
|--------|---------|--------|----|
| Régression Logistique | 0.71 | 0.62 | 0.58 |
| Random Forest | 0.79 | 0.71 | 0.68 |
| LSTM simple | 0.84 | 0.79 | 0.76 |
| LSTM Bidi | 0.88 | 0.83 | 0.81 |
| **LSTM Bidi + Attention** | **0.91** | **0.87** | **0.85** |

---

## 📦 Installation

### Backend (Python / FastAPI)

```bash
# Cloner le repo
git clone https://github.com/votre-username/EarlyAlert.git
cd EarlyAlert

# Installer les dépendances Python
pip install -r requirements.txt

# Lancer l'API FastAPI
uvicorn api.main:app --reload --port 8000
# Swagger UI disponible sur http://localhost:8000/docs
```

### Frontend (React)

```bash
cd frontend

# Installer les dépendances Node
npm install

# Lancer en développement
npm run dev
# Interface disponible sur http://localhost:5173

# Build de production
npm run build
```

### requirements.txt
```
tensorflow==2.13.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
fastapi==0.103.0
uvicorn==0.23.2
pydantic==2.3.0
python-multipart==0.0.6
```

### frontend/package.json (dépendances clés)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "recharts": "^2.8.0",
    "tailwindcss": "^3.3.0",
    "@shadcn/ui": "latest",
    "lucide-react": "^0.383.0",
    "axios": "^1.5.0"
  },
  "devDependencies": {
    "vite": "^4.4.0",
    "typescript": "^5.1.0",
    "@types/react": "^18.2.0"
  }
}
```

---

## 🐳 Docker

```bash
# Lancer frontend + backend ensemble
docker-compose up --build

# Frontend → http://localhost:3000
# API      → http://localhost:8000
# Swagger  → http://localhost:8000/docs
```

```yaml
# docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    command: uvicorn api.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports: ["3000:80"]
    depends_on: [api]
```

---

## 🎤 Pitch Jury (45 secondes)

> *"J'ai entraîné from scratch un LSTM bidirectionnel avec mécanisme d'attention  
> sur 300 000 séjours en réanimation réels issus de MIMIC-IV.  
> Mon modèle atteint une AUC de 0.91 et détecte 87% des détériorations critiques  
> 4 à 6 heures avant qu'elles soient visibles cliniquement.  
> Le mécanisme d'attention permet d'expliquer au médecin quel signal a déclenché l'alerte —  
> ce n'est pas une boîte noire. Le système est déployé et démontrable en temps réel,  
> avec un dashboard React connecté en temps réel à une API FastAPI."*

---

## 🤖 PROMPT LLM — Continuer le Projet EarlyAlert

> Copie-colle ce prompt dans n'importe quel LLM (Claude, ChatGPT, Gemini)  
> pour obtenir une aide détaillée et pédagogique sur chaque partie du projet.

```
Tu es un expert en Deep Learning médical, séries temporelles cliniques, et développement fullstack.
Je travaille sur un projet appelé EarlyAlert — un système de détection précoce
de détérioration de patients en réanimation, basé sur un LSTM Bidirectionnel
avec mécanisme d'Attention de Bahdanau, entraîné from scratch sur MIMIC-IV.
L'interface est un dashboard React connecté à une API FastAPI.

Voici le contexte complet du projet :

OBJECTIF : Prédire une détérioration critique 4 à 6 heures avant qu'elle
soit cliniquement visible, en analysant les séries temporelles de constantes
vitales (FC, TA, SpO2, FR, Température, Diurèse).

STACK TECHNIQUE :
- Python 3.10 + FastAPI (backend / API REST)
- TensorFlow / Keras (modèle DL)
- Pandas, NumPy (traitement données)
- Scikit-learn (prétraitement, métriques, baselines)
- KNNImputer (valeurs manquantes)
- Matplotlib, Seaborn (évaluation)
- React 18 + Vite + TypeScript (frontend)
- Recharts (graphiques interactifs)
- shadcn/ui + Tailwind CSS (UI)
- Dataset : MIMIC-IV (PhysioNet)
- Environnement : Kaggle GPU T4

ARCHITECTURE DU MODÈLE :
Input → Bidirectional LSTM(64, return_sequences=True) → LayerNorm → Dropout(0.3)
→ Bidirectional LSTM(32, return_sequences=True) → Dropout(0.2)
→ Attention de Bahdanau (Dense tanh + Softmax + Multiply)
→ Flatten → Dense(64, relu) → Dropout(0.3) → Dense(32, relu)
→ Dense(1, sigmoid) → Probabilité de détérioration

API FASTAPI :
- POST /predict → reçoit { sequence: float[][] } → retourne { risk_score: float, attention_weights: float[] }
- GET /patient/{id} → retourne les dernières constantes vitales

DONNÉES :
- Fenêtres glissantes de 24 timesteps × 10 features (= 6h à 15min d'intervalle)
- Label = 1 si détérioration dans les 6h suivantes, 0 sinon
- Déséquilibre : ~15% de positifs → class_weight utilisé
- Split : 70% train / 15% val / 15% test

ÉTAPE ACTUELLE DU PROJET : [INDIQUE ICI OÙ TU EN ES]
Exemples :
- "Je commence le prétraitement des données MIMIC-IV"
- "J'entraîne le modèle mais le recall est trop faible"
- "Je veux construire l'endpoint FastAPI /predict"
- "Je veux construire le composant React AttentionHeatmap"
- "Je veux connecter le frontend React à l'API FastAPI"

MA QUESTION : [ÉCRIS TA QUESTION ICI]

IMPORTANT — Comment tu dois répondre :
1. Explique chaque partie du code ligne par ligne avec son rôle clinique et technique
2. Justifie chaque choix architectural (pourquoi LSTM bidi ? pourquoi 24 timesteps ?)
3. Donne des exemples concrets avec des valeurs médicales réelles
4. Si tu donnes du code, commente chaque bloc avec son objectif
5. Si je fais une erreur, explique pourquoi c'est une erreur et comment la corriger
6. Relie toujours les décisions techniques à leur impact médical réel
7. Propose des pistes d'amélioration si pertinent
```

---

*Projet réalisé dans le cadre d'une candidature passerelle médecine*  
*Stack : TensorFlow · MIMIC-IV · FastAPI · React · Docker*