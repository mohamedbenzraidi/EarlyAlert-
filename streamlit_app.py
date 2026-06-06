import streamlit as st
import pandas as pd
from src.predict import predict_deterioration

st.set_page_config(page_title="EarlyAlert ICU", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inconsolata:wght@300;400;500;600;700&display=swap');

:root {
    --black:     #080A08;
    --panel:     #0D100D;
    --border:    #1C251C;
    --green:     #00FF87;
    --green-dim: #00C264;
    --green-glow:rgba(0,255,135,0.12);
    --amber:     #FFB800;
    --red:       #FF3B3B;
    --white:     #E8EDE8;
    --muted:     #4A5E4A;
    --display:   'Bebas Neue', cursive;
    --mono:      'Inconsolata', monospace;
}

html, body, [class*="css"] {
    background-color: var(--black) !important;
    color: var(--white) !important;
    font-family: var(--mono) !important;
}

body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,135,0.015) 2px, rgba(0,255,135,0.015) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2.5rem 4rem 2.5rem !important; max-width: 1200px !important; }

/* HEADER */
.icu-header {
    border-bottom: 1px solid var(--border);
    padding: 1.6rem 0 1.2rem;
    margin-bottom: 2.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}
.icu-logo {
    font-family: var(--display);
    font-size: 3.8rem;
    line-height: 0.9;
    letter-spacing: 0.04em;
    color: var(--green);
    text-shadow: 0 0 40px rgba(0,255,135,0.4), 0 0 80px rgba(0,255,135,0.15);
    animation: pulse-logo 4s ease-in-out infinite;
}
@keyframes pulse-logo {
    0%,100% { text-shadow: 0 0 40px rgba(0,255,135,0.4), 0 0 80px rgba(0,255,135,0.15); }
    50%      { text-shadow: 0 0 60px rgba(0,255,135,0.6), 0 0 120px rgba(0,255,135,0.25); }
}
.icu-logo span { color: var(--white); }
.icu-sub {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 6px;
}
.icu-meta {
    text-align: right;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    line-height: 1.9;
}
.icu-meta strong { color: var(--green-dim); }
.pulse-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 8px var(--green);
    animation: blink 1.4s step-end infinite;
    vertical-align: middle;
    margin-right: 6px;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

/* SECTION LABEL */
.s-label {
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 0 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    padding: 1.4rem !important;
    transition: border-color .3s, box-shadow .3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--green-dim) !important;
    box-shadow: 0 0 20px var(--green-glow) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stFileUploader"] button {
    background: transparent !important;
    border: 1px solid var(--green-dim) !important;
    color: var(--green) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 1px !important;
    transition: all .2s;
}
[data-testid="stFileUploader"] button:hover {
    background: var(--green-glow) !important;
    box-shadow: 0 0 12px var(--green-glow) !important;
}



/* METRICS */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-top: 2px solid var(--green) !important;
    border-radius: 2px !important;
    padding: 1.4rem 1.6rem !important;
    position: relative; overflow: hidden;
}
[data-testid="stMetric"]::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at top left, var(--green-glow), transparent 60%);
    pointer-events: none;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--display) !important;
    font-size: 3rem !important;
    color: var(--green) !important;
    text-shadow: 0 0 20px rgba(0,255,135,0.3) !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
}

/* RISK BLOCK */
.risk-block {
    margin: 2rem 0;
    padding: 2rem 2.4rem;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 2px;
    position: relative; overflow: hidden;
}
.risk-block::before {
    content: 'RISK INDEX';
    position: absolute;
    top: 1.2rem; right: 1.6rem;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: var(--muted);
}
.risk-number {
    font-family: var(--display);
    font-size: 7rem;
    line-height: 0.85;
}
.risk-bar-outer {
    width: 100%; height: 4px;
    background: #1C251C;
    margin: 1.2rem 0 0.5rem;
    position: relative; border-radius: 1px;
}
.risk-bar-inner {
    height: 100%; border-radius: 1px;
    transition: width 1s cubic-bezier(0.16,1,0.3,1);
}
.risk-zones {
    display: flex;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    justify-content: space-between;
    margin-top: 4px;
}
.risk-tag {
    display: inline-block;
    font-family: var(--display);
    font-size: 1.8rem;
    letter-spacing: 0.12em;
    padding: 2px 16px 0;
    border: 1px solid;
    border-radius: 1px;
    margin-top: 1rem;
}

/* RECOMMENDATION */
.rec-panel {
    padding: 1.6rem 2rem;
    border: 1px solid;
    border-radius: 2px;
    margin-top: 1.8rem;
    position: relative;
}
.rec-panel::before {
    content: attr(data-label);
    position: absolute;
    top: -9px; left: 20px;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0 8px;
    background: var(--black);
}
.rec-panel ul {
    margin: 0.6rem 0 0; padding-left: 1.2rem;
    font-size: 0.82rem; line-height: 2; letter-spacing: 0.03em;
}
.rec-high { border-color: #FF3B3B; color: #FFAAAA; }
.rec-high::before { color: #FF3B3B; }
.rec-med  { border-color: #FFB800; color: #FFE0A0; }
.rec-med::before  { color: #FFB800; }
.rec-low  { border-color: var(--green); color: #AAFFDD; }
.rec-low::before  { color: var(--green); }

/* MESSAGES */
[data-testid="stInfo"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--green-dim) !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}
[data-testid="stError"] {
    background: var(--panel) !important;
    border-left: 3px solid #FF3B3B !important;
    border-radius: 2px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}

/* EXPANDER */
[data-testid="stExpander"] {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stJson"] { font-family: var(--mono) !important; font-size: 0.74rem !important; }

/* SPINNER */
[data-testid="stSpinner"] {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--green) !important;
    letter-spacing: 0.1em !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--mono) !important; font-size: 0.76rem !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: var(--display) !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.12em !important;
    color: var(--green) !important;
    margin-top: 1.6rem !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #4A5E4A !important; line-height: 1.9 !important; }
[data-testid="stSidebar"] code {
    background: var(--black) !important;
    color: var(--green) !important;
    border: 1px solid var(--border) !important;
    padding: 2px 6px !important;
    border-radius: 1px !important;
}

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 2.5rem 0 !important; }

/* EMPTY STATE */
.empty-state {
    border: 1px dashed var(--border);
    border-radius: 2px;
    padding: 4rem 2rem;
    text-align: center;
    color: var(--muted);
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    line-height: 2.5;
    background: var(--panel);
    position: relative; overflow: hidden;
}
.empty-state::before {
    content: 'NO DATA';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-family: var(--display);
    font-size: 8rem;
    color: rgba(0,255,135,0.025);
    white-space: nowrap;
    letter-spacing: 0.08em;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───
st.markdown("""
<div class="icu-header">
    <div>
        <div class="icu-logo">Early<span>Alert</span></div>
        <div class="icu-sub">Détection précoce de détérioration clinique — Réanimation</div>
    </div>
    <div class="icu-meta">
        <span class="pulse-dot"></span><strong>SYSTEM ONLINE</strong><br>
        Modèle — BiLSTM + Attention Bahdanau<br>
        Horizon prédictif — <strong>6 heures</strong><br>
        AUC-ROC — <strong>0.85</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───
st.sidebar.markdown("### Format CSV")
st.sidebar.markdown("Chaque ligne = 1 timestep **(15 min)**  \nMinimum **24 lignes** = 6 heures")
with st.sidebar.expander("Colonnes requises"):
    st.markdown("""
**Vitales**  
`fc` `tas` `tad` `fr` `spo2` `temperature`

**Biomarqueurs**  
`lactate` `creatinine` `hemoglobin` `wbc`

**Démographie**  
`age` `sex_male` `on_ventilator`
    """)
st.sidebar.markdown("### Seuils")
st.sidebar.markdown("```\nLOW     0 ─── 50 %\nMEDIUM  50 ── 70 %\nHIGH    70 ── 100 %\n```")
st.sidebar.markdown("---")
st.sidebar.markdown('<p style="font-size:0.65rem; color:#2E3E2E; letter-spacing:0.06em; line-height:2;">Ce système ne remplace pas<br>l\'expertise médicale.</p>', unsafe_allow_html=True)

# ─── UPLOAD ───
st.markdown('<div class="s-label">Données patient — Chargement du fichier</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type="csv", label_visibility="collapsed")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.markdown('<div class="s-label" style="margin-top:2rem;">Aperçu — Dernières constantes</div>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        st.info(f"{len(df)} timesteps  ·  {df.shape[1]} colonnes  ·  {len(df)*15} min de données")

        st.markdown('<div class="s-label" style="margin-top:2.5rem;">Analyse en cours</div>', unsafe_allow_html=True)
        with st.spinner("Calcul de la probabilité de détérioration..."):
            result = predict_deterioration(df)

        prob = result["probability"]
        risk = result["risk_level"]
        pct  = prob * 100

        if risk == "HIGH":
            color = "#FF3B3B"; glow = "rgba(255,59,59,0.25)"; tag_cls = "rec-high"
            rec_label = "RISQUE ÉLEVÉ — ACTION IMMÉDIATE"
            rec_items = [
                "Alerte médicale immédiate",
                "Contacter l'équipe de réanimation",
                "Vérifier lactate, TA, SpO2 en urgence",
                "Préparer vasopresseurs et matériel d'intubation",
                "Monitoring continu toutes les minutes",
            ]
        elif risk == "MEDIUM":
            color = "#FFB800"; glow = "rgba(255,184,0,0.2)"; tag_cls = "rec-med"
            rec_label = "RISQUE MODÉRÉ — SURVEILLANCE ACCRUE"
            rec_items = [
                "Augmenter la fréquence de surveillance",
                "Monitorer les tendances des constantes vitales",
                "Réévaluer dans 30 à 60 minutes",
                "Consulter l'équipe médicale en cas d'aggravation",
                "Préparer les ressources pour intervention rapide",
            ]
        else:
            color = "#00FF87"; glow = "rgba(0,255,135,0.15)"; tag_cls = "rec-low"
            rec_label = "RISQUE FAIBLE — SUIVI STANDARD"
            rec_items = [
                "Suivi standard recommandé",
                "Continuer le monitoring régulier",
                "Réévaluer toutes les 2 à 4 heures",
                "Alerter en cas de changement significatif",
                "Retester à chaque mise à jour des données",
            ]

        st.markdown('<div class="s-label" style="margin-top:2.5rem;">Résultat</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Probabilité", f"{prob:.1%}")
        with c2: st.metric("Niveau de risque", risk)
        with c3: st.metric("Timesteps analysés", result["timesteps_used"])

        items_html = "".join(f"<li>{i}</li>" for i in rec_items)
        st.markdown(f"""
        <div class="risk-block" style="border-top:3px solid {color}; box-shadow:0 0 40px {glow};">
            <div class="risk-number" style="color:{color}; text-shadow:0 0 40px {glow};">
                {pct:.1f}<span style="font-size:2.5rem;color:#4A5E4A;">%</span>
            </div>
            <div class="risk-bar-outer">
                <div class="risk-bar-inner" style="width:{pct:.1f}%; background:{color}; box-shadow:0 0 8px {color};"></div>
            </div>
            <div class="risk-zones">
                <span>0%</span><span>LOW</span>
                <span>·</span><span>50%</span><span>·</span>
                <span>MEDIUM</span><span>70%</span><span>HIGH</span><span>100%</span>
            </div>
            <div class="risk-tag" style="color:{color}; border-color:{color}; text-shadow:0 0 20px {glow};">{risk}</div>
        </div>
        <div class="rec-panel {tag_cls}" data-label="{rec_label}">
            <ul>{items_html}</ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="s-label" style="margin-top:2.5rem;"></div>', unsafe_allow_html=True)
        with st.expander("Détails techniques du modèle"):
            st.json({
                "probability": f"{prob:.4f}",
                "risk_level": risk,
                "timesteps_used": result["timesteps_used"],
                "features_engineered": result["features_engineered"],
                "model": "BiLSTM Bidirectionnel + Attention Bahdanau",
                "input_shape": f"({result['timesteps_used']}, 18 features)",
                "auc_roc": "0.85",
                "sensibilite": "80%",
                "specificite": "75%",
            })

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.65rem; color:#2E3E2E; letter-spacing:0.08em; text-transform:uppercase; line-height:2;">Avertissement · Les résultats produits par ce système doivent être interprétés par des professionnels médicaux qualifiés. Ce modèle ne remplace pas le jugement clinique.</p>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erreur de traitement : {str(e)}")
        st.info("Colonnes requises : fc, tas, tad, fr, spo2, temperature, age, sex_male · valeurs numériques · minimum 24 lignes")

else:
    st.markdown("""
    <div class="empty-state">
        <svg width="100%" height="44" viewBox="0 0 700 44" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:1.5rem;display:block;">
            <polyline points="0,22 180,22 200,22 215,6 228,38 240,6 253,38 265,22 290,22 700,22"
                fill="none" stroke="#1C251C" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        Aucune donnée chargée<br>
        Sélectionnez un fichier CSV pour démarrer l'analyse
    </div>
    """, unsafe_allow_html=True)