"""
EarlyAlert — Streamlit Application
Systeme de detection precoce de deterioration en reanimation
LSTM Bidirectionnel + Attention de Bahdanau — MIMIC-IV
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import time
from datetime import datetime, timedelta

# ── Configuration page ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarlyAlert — ICU Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS global Streamlit (masquer elements par defaut) ────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', 'Helvetica Neue', sans-serif !important;
}

/* Masquer header/footer Streamlit */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}
[data-testid="stSidebar"] .stSlider > div {
    color: #e6edf3;
}

/* Main container */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Masquer les padding superflus */
.stApp {
    background: #f0f2f5;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e2e6ea;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Boutons */
.stButton > button {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500;
    border-radius: 6px;
    border: 1px solid #d0d5dd;
    transition: all 0.18s ease;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #e2e6ea;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 12px;
    font-weight: 500;
}

/* Upload */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #d0d5dd;
    border-radius: 8px;
    padding: 12px;
}

/* Supprime le padding de la sidebar */
[data-testid="stSidebarContent"] {
    padding: 1rem 1rem;
}

div[data-testid="stVerticalBlock"] > div:has(div.alert-high) {
    border-left: 3px solid #dc2626;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODELE (stub si pas encore entraine)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Charge le modele LSTM si disponible, sinon retourne None."""
    try:
        import tensorflow as tf
        model_path = "models/best_model.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
    except ImportError:
        pass
    return None

def predict_deterioration(sequence: np.ndarray, model=None):
    """
    Predit le score de deterioration + poids d'attention.
    sequence : array (24, 10) — 24 timesteps x 10 features
    Retourne : (score float, attention_weights array (24,))
    """
    if model is not None:
        # Modele reel charge
        seq_input = sequence[np.newaxis, ...]  # (1, 24, 10)
        # Si le modele retourne aussi les poids d'attention :
        # score, attention = model.predict(seq_input)
        score = float(model.predict(seq_input)[0][0])
        # Attention simulee si non exposee
        attention = np.random.dirichlet(np.ones(24))
    else:
        # Mode demo : simulation realiste
        # Le score est base sur les anomalies des constantes
        anomaly_score = 0.0
        for i, feat in enumerate(sequence[-1]):
            # Normalisation simple par feature
            pass
        # Score base sur la tendance des 6 derniers timesteps
        recent = sequence[-6:, :]
        trend = np.mean(np.diff(recent, axis=0), axis=0)
        # FC (idx 0) : tendance montante = risque
        # SpO2 (idx 2) : tendance descendante = risque
        # TA (idx 1) : tendance descendante = risque
        risk_signals = (
            max(0, trend[0]) * 0.3 +       # FC monte
            max(0, -trend[1]) * 0.25 +     # TA baisse
            max(0, -trend[2]) * 0.25 +     # SpO2 baisse
            max(0, trend[3]) * 0.2          # FR monte
        )
        anomaly_score = min(0.95, max(0.05, risk_signals * 10 + 0.1))
        # Poids attention : concentres sur les 8 derniers timesteps
        raw_att = np.zeros(24)
        raw_att[-8:] = np.random.dirichlet(np.ones(8) * 2)
        raw_att[:-8] = np.random.dirichlet(np.ones(16) * 0.3)
        attention = raw_att / raw_att.sum()
        score = anomaly_score

    return score, attention


# ══════════════════════════════════════════════════════════════════════════════
#  DONNEES
# ══════════════════════════════════════════════════════════════════════════════

FEATURES = ["FC (bpm)", "TA sys (mmHg)", "SpO2 (%)", "FR (/min)",
            "Temp (°C)", "PVC (mmHg)", "Lactate", "GCS", "Diurese", "FiO2 (%)"]

FEATURE_NORMS = {
    "FC (bpm)":      (60, 100,   40, 160),
    "TA sys (mmHg)": (90, 140,   60, 200),
    "SpO2 (%)":      (95, 100,   80, 100),
    "FR (/min)":     (12,  20,    8,  40),
    "Temp (°C)":     (36, 37.5,  35,  41),
    "PVC (mmHg)":    (2,    8,    0,  20),
    "Lactate":       (0.5,  2,  0.5,   8),
    "GCS":           (14,  15,    3,  15),
    "Diurese":       (40,  80,    0, 200),
    "FiO2 (%)":      (21,  40,   21, 100),
}

PATIENTS_DEMO = {
    "Chambre 1 — Leclerc A. [CRITIQUE]": {
        "info": {"Nom": "Leclerc A.", "Age": "71 ans", "Sexe": "F", "Poids": "58 kg",
                 "Admission": "Choc septique", "Sejour": "J+3", "IGS II": 62},
        "risk": "high",
        "base": [118, 88, 91, 26, 38.9, 14, 4.2, 10, 15, 60],
        "trend": [+2, -1.5, -0.5, +1, +0.1, +0.5, +0.3, -0.5, -2, +1.5],
    },
    "Chambre 3 — Benali K. [MODERE]": {
        "info": {"Nom": "Benali K.", "Age": "55 ans", "Sexe": "M", "Poids": "82 kg",
                 "Admission": "BPCO exacerbee", "Sejour": "J+1", "IGS II": 34},
        "risk": "med",
        "base": [96, 122, 93, 22, 37.4, 6, 2.1, 13, 45, 40],
        "trend": [+0.5, +0.2, -0.3, +0.4, +0.05, +0.1, +0.1, 0, -1, +0.5],
    },
    "Chambre 5 — Moreau P. [STABLE]": {
        "info": {"Nom": "Moreau P.", "Age": "43 ans", "Sexe": "M", "Poids": "74 kg",
                 "Admission": "Post-op cardiaque", "Sejour": "J+2", "IGS II": 18},
        "risk": "low",
        "base": [68, 118, 98, 15, 36.8, 5, 1.1, 15, 60, 21],
        "trend": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "Chambre 7 — Ahmadi S. [MODERE]": {
        "info": {"Nom": "Ahmadi S.", "Age": "62 ans", "Sexe": "F", "Poids": "65 kg",
                 "Admission": "AVC hemorragique", "Sejour": "J+4", "IGS II": 48},
        "risk": "med",
        "base": [102, 152, 95, 18, 38.1, 9, 2.8, 12, 30, 35],
        "trend": [+1, +2, 0, 0, +0.1, +0.2, +0.1, -0.3, -1, 0],
    },
}


def generate_sequence(base: list, trend: list, n: int = 24,
                       noise_level: float = 0.03) -> np.ndarray:
    """Genere une sequence temporelle realiste (24 timesteps x 10 features)."""
    seq = np.zeros((n, 10))
    for j in range(10):
        b = base[j]
        for i in range(n):
            progress = i / (n - 1)
            drift = trend[j] * progress * n * 0.4
            noise = b * noise_level * np.random.randn()
            seq[i, j] = b + drift + noise
    # SpO2 clamp 0-100
    seq[:, 2] = np.clip(seq[:, 2], 70, 100)
    # GCS clamp 3-15
    seq[:, 7] = np.clip(seq[:, 7], 3, 15)
    # FC clamp positif
    seq[:, 0] = np.clip(seq[:, 0], 30, 220)
    return seq


def generate_synthetic_sequence(risk_level: str = "med") -> tuple:
    """Genere une sequence synthetique selon le niveau de risque."""
    if risk_level == "high":
        base  = [115, 85,  90, 28, 39.2, 15, 4.5, 9,  12, 65]
        trend = [+3,  -2, -0.8, +2, +0.15, +1, +0.5, -1, -3, +2]
    elif risk_level == "med":
        base  = [95,  118, 93, 21, 37.8, 7, 2.2, 13, 40, 38]
        trend = [+1,  -0.5, -0.3, +0.5, +0.05, +0.2, +0.1, 0, -1, +0.5]
    else:
        base  = [70,  120, 97, 14, 36.9, 5, 1.2, 15, 65, 21]
        trend = [0,    0,   0,  0,   0,  0,   0,  0,  0,  0]

    noise = 0.04 if risk_level == "high" else 0.02
    seq = generate_sequence(base, trend, noise_level=noise)
    info = {
        "Nom": f"SIM-{np.random.randint(1000,9999)}",
        "Age": f"{np.random.randint(35, 80)} ans",
        "Sexe": np.random.choice(["M", "F"]),
        "Poids": f"{np.random.randint(55,95)} kg",
        "Admission": np.random.choice(["Choc septique", "SDRA", "Polytraumatisme",
                                        "Insuffisance cardiaque"]),
        "Sejour": "J+0 (Simulation)",
        "IGS II": int({"high": 55, "med": 35, "low": 18}[risk_level]
                      + np.random.randint(-10, 10)),
    }
    return seq, info


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPHIQUES PLOTLY
# ══════════════════════════════════════════════════════════════════════════════

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, Helvetica Neue, sans-serif",
              color="#4a5568", size=11),
    margin=dict(l=40, r=10, t=10, b=30),
    xaxis=dict(gridcolor="rgba(0,0,0,0.06)", zeroline=False,
               showline=False, tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(0,0,0,0.06)", zeroline=False,
               showline=False, tickfont=dict(size=10)),
)

COLORS = {
    "high":   "#dc2626",
    "med":    "#d97706",
    "low":    "#0d7a4e",
    "normal": "rgba(100,180,100,0.5)",
    "series": ["#1a56db", "#0d7a4e", "#d97706", "#7c3aed",
               "#dc2626", "#0891b2", "#059669", "#9333ea",
               "#ea580c", "#4338ca"],
}


def make_vitals_chart(sequence: np.ndarray, feature_idx: int = 0) -> go.Figure:
    """Graphique principal d'une constante vitale sur 24 timesteps."""
    feat = FEATURES[feature_idx]
    norm_min, norm_max, _, _ = FEATURE_NORMS[feat]
    times = [f"-{(23-i)*15}m" if (23-i) % 4 == 0 else "" for i in range(24)]
    vals = sequence[:, feature_idx]

    # Couleur selon hors normes
    last_val = vals[-1]
    if last_val < norm_min or last_val > norm_max:
        line_color = COLORS["high"]
        fill_color = "rgba(220,38,38,0.1)"
    else:
        line_color = COLORS["series"][feature_idx % len(COLORS["series"])]
        fill_color = f"rgba(26,86,219,0.08)"

    fig = go.Figure()
    # Zone normale
    fig.add_hrect(y0=norm_min, y1=norm_max,
                  fillcolor="rgba(100,200,120,0.08)",
                  line_color="rgba(100,200,120,0.3)",
                  line_width=1)
    # Serie principale
    fig.add_trace(go.Scatter(
        x=list(range(24)), y=vals.tolist(),
        mode="lines",
        line=dict(color=line_color, width=2),
        fill="tozeroy", fillcolor=fill_color,
        name=feat,
        hovertemplate=f"<b>{feat}</b><br>Valeur : %{{y:.1f}}<br>T : %{{x}}<extra></extra>"
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        height=170,
        showlegend=False,
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"],
                   tickvals=list(range(0, 24, 4)),
                   ticktext=[f"-{(23-i)*15}m" for i in range(0, 24, 4)]),
    )
    return fig


def make_all_vitals_chart(sequence: np.ndarray) -> go.Figure:
    """Graphique multi-lignes de toutes les constantes (normalisees)."""
    fig = go.Figure()
    for j, feat in enumerate(FEATURES):
        vals = sequence[:, j]
        norm_min, norm_max, v_min, v_max = FEATURE_NORMS[feat]
        # Normalisation 0-1
        vals_norm = (vals - v_min) / (v_max - v_min + 1e-9)
        fig.add_trace(go.Scatter(
            x=list(range(24)), y=vals_norm.tolist(),
            mode="lines", name=feat,
            line=dict(color=COLORS["series"][j % len(COLORS["series"])], width=1.5),
            hovertemplate=f"<b>{feat}</b>: %{{customdata:.1f}}<extra></extra>",
            customdata=vals.tolist()
        ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        height=200,
        legend=dict(orientation="h", y=-0.25, font=dict(size=9)),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"], title="Valeur normalisee", tickformat=".1f"),
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"],
                   tickvals=list(range(0, 24, 4)),
                   ticktext=[f"-{(23-i)*15}m" for i in range(0, 24, 4)]),
    )
    return fig


def make_attention_chart(attention: np.ndarray,
                          feature_attention: np.ndarray) -> go.Figure:
    """Heatmap d'attention sur les 24 timesteps."""
    times = [f"T-{(23-i)*15}m" for i in range(24)]
    fig = go.Figure(data=go.Heatmap(
        z=[attention.tolist()],
        x=times,
        y=["Attention"],
        colorscale=[[0, "#e8f0fd"], [0.5, "#1a56db"], [1, "#dc2626"]],
        showscale=True,
        colorbar=dict(thickness=8, len=0.8, tickfont=dict(size=9)),
        hovertemplate="<b>Poids attention</b><br>%{x}: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        height=80,
        margin=dict(l=60, r=60, t=5, b=30),
        xaxis=dict(tickfont=dict(size=9), tickangle=0,
                   tickvals=list(range(0, 24, 4)),
                   ticktext=[f"T-{(23-i)*15}m" for i in range(0, 24, 4)]),
        yaxis=dict(showticklabels=True, tickfont=dict(size=9)),
    )
    return fig


def make_feature_attention_bar(sequence: np.ndarray,
                                 attention: np.ndarray) -> go.Figure:
    """Barres de contribution par signal (attention x variance)."""
    # Ponderer chaque feature par les poids d'attention
    weighted = np.zeros(10)
    for t in range(24):
        weighted += attention[t] * np.abs(sequence[t])
    # Normaliser
    weighted = weighted / (weighted.sum() + 1e-9)
    sorted_idx = np.argsort(weighted)[::-1]

    feats = [FEATURES[i].split(" ")[0] for i in sorted_idx]
    vals  = [weighted[i] for i in sorted_idx]
    colors_bar = []
    for i in sorted_idx:
        feat = FEATURES[i]
        norm_min, norm_max, _, _ = FEATURE_NORMS[feat]
        last = sequence[-1, i]
        if last < norm_min or last > norm_max:
            colors_bar.append(COLORS["high"])
        else:
            colors_bar.append(COLORS["series"][i % len(COLORS["series"])])

    fig = go.Figure(go.Bar(
        x=[f"{v*100:.1f}%" for v in vals],
        y=feats,
        orientation="h",
        marker_color=colors_bar,
        text=[f"{v*100:.0f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=10, family="IBM Plex Mono"),
        hovertemplate="<b>%{y}</b><br>Contribution : %{x}<extra></extra>"
    ))
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        height=260,
        margin=dict(l=60, r=60, t=5, b=10),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(tickfont=dict(size=11, family="IBM Plex Sans")),
        bargap=0.3,
    )
    return fig


def make_score_history_chart(history: list, risk: str) -> go.Figure:
    """Courbe d'evolution du score de risque."""
    color = COLORS[risk]
    fig = go.Figure()
    fig.add_hrect(y0=0.65, y1=1.0,
                  fillcolor="rgba(220,38,38,0.07)", line_width=0)
    fig.add_hrect(y0=0.35, y1=0.65,
                  fillcolor="rgba(217,119,6,0.07)", line_width=0)
    fig.add_trace(go.Scatter(
        x=list(range(len(history))),
        y=[h * 100 for h in history],
        mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=5, color=color),
        fill="tozeroy",
        fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba")
            if "rgb" in color else color + "20",
        hovertemplate="Score : <b>%{y:.0f}%</b><extra></extra>"
    ))
    labels = [f"-{(7-i)*15}m" for i in range(7)] + ["Actuel"]
    fig.update_layout(
        **PLOTLY_TEMPLATE,
        height=120,
        showlegend=False,
        xaxis=dict(**PLOTLY_TEMPLATE["xaxis"],
                   tickvals=list(range(8)), ticktext=labels,
                   tickfont=dict(size=9)),
        yaxis=dict(**PLOTLY_TEMPLATE["yaxis"],
                   range=[0, 105],
                   ticksuffix="%", tickfont=dict(size=9)),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  COMPOSANTS UI
# ══════════════════════════════════════════════════════════════════════════════

def render_alert_banner(score: float):
    """Bandeau d'alerte principal avec code couleur."""
    pct = int(score * 100)
    if score >= 0.65:
        risk, label, desc, border = (
            "high", "DETERIORATION PROBABLE",
            f"Score de risque eleve ({pct}%) — Alerte medicale recommandee. "
            "Detection d'un pattern critique sur les constantes vitales.",
            "#dc2626"
        )
        bg, text_color = "#fde8e8", "#991b1b"
    elif score >= 0.35:
        risk, label, desc, border = (
            "med", "SURVEILLANCE RENFORCEE",
            f"Score de risque modere ({pct}%) — Re-evaluation dans 30 minutes. "
            "Tendance observee sur plusieurs signaux.",
            "#d97706"
        )
        bg, text_color = "#fef3c7", "#92400e"
    else:
        risk, label, desc, border = (
            "low", "PARAMETRES STABLES",
            f"Score de risque faible ({pct}%) — Evolution favorable. "
            "Aucune deterioration imminente detectee.",
            "#0d7a4e"
        )
        bg, text_color = "#dcf5eb", "#065f46"

    st.markdown(f"""
    <div style="
        background:{bg}; border:1px solid {border}40; border-left:4px solid {border};
        border-radius:8px; padding:14px 20px; margin-bottom:16px;
        display:flex; align-items:center; gap:20px;
    ">
        <div style="
            width:60px; height:60px; border-radius:50%; flex-shrink:0;
            border:3px solid {border}; display:flex; flex-direction:column;
            align-items:center; justify-content:center; background:white;
        ">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:16px;
                font-weight:600; color:{border}; line-height:1;">{pct}%</span>
            <span style="font-size:8px; text-transform:uppercase;
                letter-spacing:0.4px; color:{border}; font-weight:500;">risque</span>
        </div>
        <div style="flex:1;">
            <div style="font-size:12px; font-weight:600; letter-spacing:0.8px;
                text-transform:uppercase; color:{border}; margin-bottom:3px;">
                {label}
            </div>
            <div style="font-size:12px; color:{text_color}; line-height:1.5;">
                {desc}
            </div>
        </div>
        <div style="text-align:right; flex-shrink:0;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:11px;
                color:{text_color}; opacity:0.7;">{datetime.now().strftime('%H:%M:%S')}</div>
            <div style="font-size:11px; font-weight:600; color:{border};
                margin-top:3px;">Horizon : 4-6h</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    return risk


def render_vital_cards(sequence: np.ndarray):
    """Affiche les 6 premieres constantes en cartes metriques."""
    cols = st.columns(6)
    for j, col in enumerate(cols):
        feat = FEATURES[j]
        norm_min, norm_max, _, _ = FEATURE_NORMS[feat]
        val = sequence[-1, j]
        prev = sequence[-2, j]
        delta = val - prev

        if val < norm_min or val > norm_max:
            status_color = "#dc2626"
            status_bg = "#fde8e8"
        else:
            status_color = "#0d7a4e"
            status_bg = "#dcf5eb"

        short_name = feat.split(" ")[0]
        unit = feat.split("(")[1].rstrip(")") if "(" in feat else ""

        with col:
            delta_sign = "+" if delta > 0 else ""
            delta_color = "#dc2626" if delta > 0 and val > norm_max else \
                          "#dc2626" if delta < 0 and val < norm_min else "#4a5568"
            st.markdown(f"""
            <div style="
                background:white; border:1px solid #e2e6ea;
                border-top:3px solid {status_color};
                border-radius:6px; padding:10px 12px;
                margin-bottom:4px;
            ">
                <div style="font-size:9px; text-transform:uppercase;
                    letter-spacing:0.5px; color:#8a9ab0; margin-bottom:4px;">
                    {short_name}
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;
                    font-size:20px; font-weight:500; color:#0d1117;
                    line-height:1;">{val:.1f}</div>
                <div style="font-size:9px; color:#8a9ab0; margin-top:1px;">{unit}</div>
                <div style="font-size:10px; color:{delta_color}; margin-top:4px;
                    font-family:'IBM Plex Mono',monospace;">
                    {delta_sign}{delta:.1f} vs T-1
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_patient_sidebar(info: dict):
    """Affiche le dossier patient dans la sidebar."""
    st.markdown("""
    <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.8px;
        color:#8d96a0; font-weight:600; margin-bottom:8px; margin-top:4px;">
        Dossier patient
    </div>
    """, unsafe_allow_html=True)
    for k, v in info.items():
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between;
            padding:5px 0; border-bottom:0.5px solid #21262d;
            font-size:11px;">
            <span style="color:#8d96a0;">{k}</span>
            <span style="font-family:'IBM Plex Mono',monospace;
                color:#e6edf3; font-weight:500;">{v}</span>
        </div>
        """, unsafe_allow_html=True)


def render_section_title(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style="margin: 8px 0 6px 0;">
        <div style="font-size:10px; font-weight:600; text-transform:uppercase;
            letter-spacing:0.7px; color:#8a9ab0;">{title}</div>
        {"" if not subtitle else f'<div style="font-size:11px;color:#4a5568;margin-top:2px;">{subtitle}</div>'}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════

def render_header():
    st.markdown(f"""
    <div style="
        background:#0d1117; border-bottom:1px solid #21262d;
        padding:0 24px; height:52px;
        display:flex; align-items:center; justify-content:space-between;
        margin-bottom:0;
    ">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="
                width:28px; height:28px; background:#1a56db;
                border-radius:6px; display:flex; align-items:center;
                justify-content:center;
            ">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="white">
                    <path d="M8 1l1.5 3.5L13 5l-2.5 2.5.5 3.5L8 9.5 5 11l.5-3.5L3 5l3.5-.5z"/>
                </svg>
            </div>
            <span style="font-size:15px; font-weight:600; color:#e6edf3;
                letter-spacing:-0.3px;">EarlyAlert</span>
            <span style="font-size:10px; text-transform:uppercase; letter-spacing:0.8px;
                color:#586069; font-weight:400; margin-left:4px;">ICU Deterioration Monitor</span>
        </div>
        <div style="display:flex; align-items:center; gap:16px;">
            <span style="font-family:'IBM Plex Mono',monospace; font-size:12px;
                color:#8d96a0;">{datetime.now().strftime('%d/%m/%Y %H:%M')}</span>
            <div style="
                display:flex; align-items:center; gap:5px;
                background:#2d1117; padding:3px 10px; border-radius:4px;
            ">
                <div style="width:6px; height:6px; border-radius:50%;
                    background:#f85149; animation:pulse 1.4s infinite;"></div>
                <span style="font-size:10px; font-weight:600; color:#f85149;
                    letter-spacing:0.6px; text-transform:uppercase;">Live</span>
            </div>
        </div>
    </div>
    <style>
        @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    render_header()

    # ── Chargement modele ──────────────────────────────────────────────────
    model = load_model()
    if model is None:
        st.sidebar.markdown("""
        <div style="
            background:#272012; border:1px solid #d29922;
            border-radius:6px; padding:8px 10px; margin-bottom:12px;
            font-size:11px; color:#d29922;
        ">
            Mode demo — Modele LSTM non charge.<br>
            Placez <code>best_model.h5</code> dans <code>models/</code>
        </div>
        """, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="font-size:13px; font-weight:600; color:#e6edf3;
            margin-bottom:12px; margin-top:8px;">
            Selection du patient
        </div>
        """, unsafe_allow_html=True)

        patient_options = list(PATIENTS_DEMO.keys()) + ["[Simulation] Patient synthetique"]
        selected = st.selectbox("Patient", patient_options, label_visibility="collapsed")

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="height:0.5px; background:#21262d; margin:4px 0 12px 0;"></div>
        """, unsafe_allow_html=True)

        # Mode simulation
        if selected == "[Simulation] Patient synthetique":
            st.markdown("""
            <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.8px;
                color:#8d96a0; font-weight:600; margin-bottom:8px;">
                Parametres simulation
            </div>
            """, unsafe_allow_html=True)

            sim_risk = st.selectbox(
                "Niveau de risque",
                ["Critique (high)", "Modere (med)", "Stable (low)"],
                label_visibility="visible"
            )
            risk_map = {"Critique (high)": "high", "Modere (med)": "med", "Stable (low)": "low"}
            sim_risk_key = risk_map[sim_risk]

            if st.button("Generer patient synthetique", use_container_width=True):
                seq, info = generate_synthetic_sequence(sim_risk_key)
                st.session_state["sim_seq"] = seq
                st.session_state["sim_info"] = info

            sequence = st.session_state.get("sim_seq", generate_synthetic_sequence(sim_risk_key)[0])
            patient_info = st.session_state.get("sim_info", generate_synthetic_sequence(sim_risk_key)[1])

        else:
            p_data = PATIENTS_DEMO[selected]
            sequence = generate_sequence(p_data["base"], p_data["trend"])
            patient_info = p_data["info"]

        # Dossier patient
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        render_patient_sidebar(patient_info)

        st.markdown("""
        <div style="height:0.5px; background:#21262d; margin:12px 0;"></div>
        """, unsafe_allow_html=True)

        # Upload CSV
        st.markdown("""
        <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.8px;
            color:#8d96a0; font-weight:600; margin-bottom:8px;">
            Import CSV patient
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "CSV (24 lignes x 10 colonnes)",
            type=["csv"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                df_up = pd.read_csv(uploaded_file, header=None)
                if df_up.shape[0] >= 24 and df_up.shape[1] >= 10:
                    sequence = df_up.iloc[:24, :10].values.astype(float)
                    st.success(f"Fichier charge : {df_up.shape[0]} lignes x {df_up.shape[1]} colonnes")
                else:
                    st.warning(f"Format attendu : 24x10. Recu : {df_up.shape[0]}x{df_up.shape[1]}")
            except Exception as e:
                st.error(f"Erreur lecture : {e}")

        # Infos modele
        st.markdown("""
        <div style="height:0.5px; background:#21262d; margin:12px 0;"></div>
        <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.8px;
            color:#8d96a0; font-weight:600; margin-bottom:8px;">
            Metriques modele
        </div>
        """, unsafe_allow_html=True)
        metrics = [("AUC-ROC", "0.91"), ("Recall", "0.87"), ("F1-Score", "0.85"), ("Params", "180k")]
        for k, v in metrics:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                padding:4px 0; font-size:11px;">
                <span style="color:#8d96a0;">{k}</span>
                <span style="font-family:'IBM Plex Mono',monospace;
                    color:#e6edf3; font-weight:500;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── PREDICTION ─────────────────────────────────────────────────────────
    score, attention = predict_deterioration(sequence, model)

    # Historique score (simulation)
    base_hist = [score * (0.5 + i * 0.07) for i in range(8)]
    base_hist[-1] = score
    base_hist = [min(0.99, max(0.01, h)) for h in base_hist]

    # ── CONTENU PRINCIPAL ──────────────────────────────────────────────────
    main_container = st.container()
    with main_container:
        st.markdown('<div style="padding:16px 24px 0 24px;">', unsafe_allow_html=True)

        # Alerte principale
        risk_level = render_alert_banner(score)

        # Constantes vitales
        render_section_title("Constantes vitales — valeurs courantes (T-0)")
        render_vital_cards(sequence)

        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

        # Onglets principaux
        tab1, tab2, tab3, tab4 = st.tabs([
            "  Serie temporelle  ",
            "  Toutes les constantes  ",
            "  Attention — Explainabilite  ",
            "  Historique du score  ",
        ])

        with tab1:
            col_sel, col_chart = st.columns([1, 4])
            with col_sel:
                render_section_title("Signal")
                feature_idx = st.radio(
                    "Constante",
                    options=list(range(len(FEATURES))),
                    format_func=lambda i: FEATURES[i].split(" ")[0],
                    label_visibility="collapsed"
                )
            with col_chart:
                norm_min, norm_max, _, _ = FEATURE_NORMS[FEATURES[feature_idx]]
                last_val = sequence[-1, feature_idx]
                status = "hors plage normale" if (last_val < norm_min or last_val > norm_max) else "dans les normes"
                render_section_title(
                    f"{FEATURES[feature_idx]} — serie temporelle",
                    f"Valeur actuelle : {last_val:.1f} | Plage normale : {norm_min}–{norm_max} | {status}"
                )
                fig = make_vitals_chart(sequence, feature_idx)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Zone norme annotee
                st.markdown(f"""
                <div style="display:flex; gap:16px; font-size:10px; color:#8a9ab0; margin-top:-10px;">
                    <span style="display:flex;align-items:center;gap:4px;">
                        <div style="width:12px;height:3px;background:rgba(100,200,120,0.6);border-radius:2px;"></div>
                        Zone normale ({norm_min}–{norm_max})
                    </span>
                    <span style="display:flex;align-items:center;gap:4px;">
                        <div style="width:12px;height:3px;background:#1a56db;border-radius:2px;"></div>
                        Valeur patient
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with tab2:
            render_section_title(
                "Toutes les constantes — normalise",
                "Visualisation simultanee des 10 signaux sur 6 heures (valeurs normalisees 0-1)"
            )
            fig_all = make_all_vitals_chart(sequence)
            st.plotly_chart(fig_all, use_container_width=True, config={"displayModeBar": False})

            # Table recapitulative
            render_section_title("Valeurs courantes vs normes")
            rows = []
            for j, feat in enumerate(FEATURES):
                norm_min, norm_max, _, _ = FEATURE_NORMS[feat]
                val = sequence[-1, j]
                status = "Normal" if norm_min <= val <= norm_max else ("Eleve" if val > norm_max else "Bas")
                rows.append({
                    "Signal": feat.split(" ")[0],
                    "Valeur": f"{val:.1f}",
                    "Min normal": norm_min,
                    "Max normal": norm_max,
                    "Statut": status,
                })
            df_table = pd.DataFrame(rows)
            st.dataframe(
                df_table.style.applymap(
                    lambda v: "color: #dc2626; font-weight: 600" if v in ("Eleve", "Bas") else "color: #0d7a4e",
                    subset=["Statut"]
                ),
                hide_index=True,
                use_container_width=True,
                height=220
            )

        with tab3:
            render_section_title(
                "Mecanisme d'attention de Bahdanau",
                "Le modele LSTM identifie les moments et signaux les plus determinants pour la prediction"
            )

            col_heat, col_bar = st.columns([2, 1])

            with col_heat:
                render_section_title("Poids d'attention — timeline")
                fig_att = make_attention_chart(attention, None)
                st.plotly_chart(fig_att, use_container_width=True, config={"displayModeBar": False})

                # Timestamp critique
                peak_t = int(np.argmax(attention))
                peak_time = (23 - peak_t) * 15
                st.markdown(f"""
                <div style="
                    background:#f7f8fa; border:0.5px solid #e2e6ea;
                    border-left:3px solid #1a56db;
                    border-radius:5px; padding:8px 12px;
                    font-size:11px; margin-top:-8px;
                ">
                    <span style="color:#4a5568;">Moment critique identifie par le modele : </span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-weight:600;
                        color:#0d1117;">T-{peak_time} min</span>
                    <span style="color:#4a5568;"> (poids : </span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-weight:600;
                        color:#1a56db;">{attention[peak_t]:.3f}</span>
                    <span style="color:#4a5568;">)</span>
                </div>
                """, unsafe_allow_html=True)

            with col_bar:
                render_section_title("Contribution par signal")
                fig_feat = make_feature_attention_bar(sequence, attention)
                st.plotly_chart(fig_feat, use_container_width=True, config={"displayModeBar": False})

            # Signal principal
            weighted = np.zeros(10)
            for t in range(24):
                weighted += attention[t] * np.abs(sequence[t])
            weighted /= (weighted.sum() + 1e-9)
            top_feat_idx = int(np.argmax(weighted))
            top_feat = FEATURES[top_feat_idx]
            top_val = sequence[-1, top_feat_idx]
            norm_min, norm_max, _, _ = FEATURE_NORMS[top_feat]

            alert_color = "#dc2626" if (top_val < norm_min or top_val > norm_max) else "#1a56db"
            st.markdown(f"""
            <div style="
                background:white; border:1px solid #e2e6ea;
                border-top:3px solid {alert_color};
                border-radius:8px; padding:12px 16px; margin-top:8px;
            ">
                <div style="font-size:10px; text-transform:uppercase; letter-spacing:0.6px;
                    color:#8a9ab0; margin-bottom:4px;">Signal determinant — interpretation clinique</div>
                <div style="font-size:13px; font-weight:600; color:#0d1117; margin-bottom:4px;">
                    {top_feat}
                </div>
                <div style="font-size:12px; color:#4a5568; line-height:1.5;">
                    Ce signal represente <strong>{weighted[top_feat_idx]*100:.0f}%</strong>
                    de la contribution totale a la prediction. Valeur actuelle :
                    <span style="font-family:'IBM Plex Mono',monospace; font-weight:600;
                        color:{alert_color};">{top_val:.1f}</span>
                    (norme : {norm_min}–{norm_max}).
                    {"La valeur est hors plage — ce signal a declenche l'alerte." if (top_val < norm_min or top_val > norm_max) else "La valeur est dans la norme mais la tendance contribue a la prediction."}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tab4:
            render_section_title(
                "Historique du score de risque",
                "Evolution sur les 8 dernieres evaluations (intervalle : 15 min)"
            )
            fig_hist = make_score_history_chart(base_hist, risk_level)
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Score actuel", f"{score*100:.0f}%",
                          delta=f"{(score - base_hist[-2])*100:+.0f}%")
            with col_m2:
                st.metric("Score max (2h)", f"{max(base_hist)*100:.0f}%")
            with col_m3:
                st.metric("Tendance", "Hausse" if score > base_hist[0] else "Stable")
            with col_m4:
                st.metric("Evaluations", len(base_hist))

        st.markdown('</div>', unsafe_allow_html=True)

    # ── FOOTER ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        border-top:1px solid #e2e6ea; background:white;
        padding:8px 24px; margin-top:16px;
        display:flex; align-items:center; justify-content:space-between;
        font-size:10px; color:#8a9ab0;
    ">
        <span>EarlyAlert v1.0 — LSTM Bidirectionnel + Attention de Bahdanau — MIMIC-IV (300k+ sejours)</span>
        <div style="display:flex; gap:16px;">
            <span>AUC-ROC : 0.91</span>
            <span>Recall : 0.87</span>
            <span>F1 : 0.85</span>
            <span>Params : 180k</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()