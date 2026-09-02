"""
theme.py
--------
Identite visuelle de l'application de segmentation.

Parti pris : palette vive de neobanque. Trois couleurs franches pour les
trois segments, un indigo electrique pour tout ce qui est interactif, et
un fond blanc lilas qui fait ressortir les aplats sans les faire vibrer.

La couleur porte de l'information : chaque segment garde la meme teinte
partout (badge, graphique, tableau), donc on reconnait un groupe a sa
couleur sans lire la legende.
"""

import streamlit as st

# =========================
# JETONS DE COULEUR
# =========================

INK = "#1E1B3A"        # texte principal (bleu nuit)
MUTED = "#6E6A8F"      # texte secondaire, legendes
BG = "#F7F5FF"         # fond
SURFACE = "#FFFFFF"    # cartes, tableaux
LINE = "#E3DEFF"       # filets, separateurs
PRIMARY = "#4B3BFF"    # indigo electrique : elements interactifs

# Un segment = une couleur, la meme dans toute l'application.
SEGMENT_COLORS = {
    "Compte de réception": "#FF5D8F",   # rose vif : le compte a reveiller
    "Compte de reception": "#FF5D8F",   # variante sans accent
    "Usage partiel":       "#FFB627",   # jaune soleil : usage a elargir
    "Compte principal":    "#00C9A7",   # turquoise : le client acquis
}

# Ordre d'affichage : du moins engage au plus engage.
SEGMENT_ORDER = ["Compte de réception", "Usage partiel", "Compte principal"]

# Palette d'appoint pour un graphique a plus de trois series.
EXTRA_COLORS = ["#4B3BFF", "#FF7A45", "#9B5DE5", "#00B4D8"]


def segment_scale(segments):
    """Retourne (domaine, gamme) pour une echelle de couleur Altair."""
    domaine = [s for s in SEGMENT_ORDER if s in set(segments)]
    gamme = [SEGMENT_COLORS.get(s, PRIMARY) for s in domaine]
    return domaine, gamme


def color_of(segment: str) -> str:
    """Couleur associee a un segment."""
    return SEGMENT_COLORS.get(segment, PRIMARY)


# =========================
# FEUILLE DE STYLE
# =========================

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stApp {{
    font-family: 'Outfit', system-ui, sans-serif;
}}

/* Chiffres alignes verticalement : indispensable des qu'on compare
   des montants d'une ligne a l'autre. */
[data-testid="stMetricValue"], table, .stDataFrame {{
    font-feature-settings: "tnum" 1, "lnum" 1;
}}

.block-container {{
    padding-top: 2.5rem;
    max-width: 1100px;
}}

h1 {{
    font-weight: 700;
    letter-spacing: -0.03em;
    color: {INK};
    margin-bottom: 0.2rem;
}}

h2, h3 {{
    font-weight: 600;
    letter-spacing: -0.015em;
    color: {INK};
}}

/* Metriques : aplat blanc franc, bord colore epais.
   Pas d'ombre grise : c'est la couleur qui structure, pas la profondeur. */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {LINE};
    border-top: 3px solid {PRIMARY};
    border-radius: 10px;
    padding: 1rem 1.2rem;
}}

[data-testid="stMetricLabel"] {{
    color: {MUTED};
    font-size: 0.8rem;
    font-weight: 500;
}}

[data-testid="stMetricValue"] {{
    color: {INK};
    font-weight: 700;
}}

/* Onglets : soulignement epais et colore sur l'onglet actif. */
.stTabs [data-baseweb="tab-list"] {{
    gap: 1.75rem;
    border-bottom: 1px solid {LINE};
}}

.stTabs [data-baseweb="tab"] {{
    padding: 0.5rem 0;
    color: {MUTED};
    font-weight: 500;
}}

.stTabs [aria-selected="true"] {{
    color: {PRIMARY};
    font-weight: 600;
}}

.stTabs [data-baseweb="tab-highlight"] {{
    background-color: {PRIMARY};
    height: 3px;
}}

/* Boutons pleins, coins arrondis, pas d'ombre. */
.stButton > button {{
    border-radius: 8px;
    border: none;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
}}

/* Encadres d'action recommandee : fond indigo tres dilue. */
[data-testid="stAlertContainer"] {{
    background: rgba(75, 59, 255, 0.06);
    border: 1px solid {LINE};
    border-left: 3px solid {PRIMARY};
    border-radius: 10px;
}}

[data-testid="stExpander"] details {{
    border: 1px solid {LINE};
    border-radius: 10px;
    background: {SURFACE};
}}

footer, #MainMenu {{ visibility: hidden; }}
</style>
"""


def apply():
    """Injecte la feuille de style. A appeler juste apres set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)


def segment_badge(segment: str, description: str = "") -> str:
    """Bandeau colore aux couleurs du segment, a passer a st.markdown."""
    couleur = color_of(segment)
    return f"""
    <div style="background:{couleur};border-radius:12px;
                padding:1.1rem 1.3rem;margin:0.6rem 0;">
        <div style="font-size:1.3rem;font-weight:700;color:#FFFFFF;
                    letter-spacing:-0.02em;">{segment}</div>
        <div style="color:rgba(255,255,255,0.92);font-size:0.92rem;
                    margin-top:0.2rem;">{description}</div>
    </div>
    """