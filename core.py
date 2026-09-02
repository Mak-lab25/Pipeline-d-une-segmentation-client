"""
core.py
-------
Logique metier partagee entre l'API FastAPI (api.py) et l'interface Streamlit (app.py).

Les transformations appliquees ici doivent rester strictement identiques a celles
de features.py : memes colonnes retenues, meme compression log, meme scaler.
Toute modification dans features.py doit etre repercutee ici.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# =========================
# CONFIGURATION
# =========================
DATA_DIR = Path(__file__).parent / "data"
if not DATA_DIR.exists():
    DATA_DIR = Path(__file__).parent / "app_data"

MODEL_PATH = DATA_DIR / "kmeans_model.pkl"
SCALER_PATH = DATA_DIR / "scaler.pkl"
COLUMNS_PATH = DATA_DIR / "feature_columns.pkl"
CLUSTERED_PATH = DATA_DIR / "df_clustered.parquet"

# Doit correspondre a CATEGORICAL_COLS dans features.py
CAT_COLUMNS = []

# Doit correspondre a DESCRIPTIVE_COLS dans features.py :
# affichees dans l'interface, absentes du modele.
DESCRIPTIVE_COLUMNS = [
    "top_merchant_country",
    "top_merchant_currency",
    "top_merchant_category",
]

# Doit correspondre a LOG_COLS dans features.py
LOG_COLUMNS = [
    "total_spent",
    "avg_transaction_amount",
    "total_transactions",
    "inbound_transactions",
    "outbound_transactions",
    "unique_merchants",
    "merchant_transactions",
]

SEGMENT_DESCRIPTIONS = {
    "Compte de réception": "Reçoit de l'argent mais paie très peu : 52 % de transactions entrantes, moins d'un commerçant régulier.",
    "Usage partiel": "Utilise le compte pour une partie de ses achats : une dizaine de commerçants, environ 40 transactions.",
    "Compte principal": "Compte du quotidien : près de 27 commerçants différents et 290 transactions en moyenne.",
}

SEGMENT_ACTIONS = {
    "Compte de réception": "Convertir en compte de paiement : carte physique, domiciliation de revenus, première offre d'achat.",
    "Usage partiel": "Élargir l'usage : cashback ciblé sur les catégories où le client paie encore ailleurs.",
    "Compte principal": "Fidéliser : offre premium et programme de fidélité, ce sont les clients à retenir en priorité.",
}


# =========================
# CHARGEMENT DES RESSOURCES
# =========================
def load_resources():
    """Charge le modele, le scaler, les colonnes attendues et les donnees segmentees."""
    missing = [
        p.name
        for p in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH, CLUSTERED_PATH)
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Artefacts manquants dans {DATA_DIR}/ : {', '.join(missing)}. "
            "Lance d'abord : python ingest.py && python features.py && python train.py"
        )

    with open(MODEL_PATH, "rb") as f:
        kmeans = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(COLUMNS_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    df_clustered = pd.read_parquet(CLUSTERED_PATH)
    return kmeans, scaler, feature_columns, df_clustered


# =========================
# FONCTIONS METIER
# =========================
def get_user_segment(df_clustered: pd.DataFrame, user_id: str) -> dict | None:
    """Retourne les infos de segment d'un utilisateur existant, ou None s'il est introuvable."""
    user_row = df_clustered[df_clustered["user_id"] == user_id]
    if user_row.empty:
        return None

    row = user_row.iloc[0]
    segment = row["segment"]
    return {
        "user_id": user_id,
        "segment": segment,
        "description": SEGMENT_DESCRIPTIONS.get(segment, ""),
        "total_spent": round(float(row["total_spent"]), 2),
        "avg_transaction": round(float(row["avg_transaction_amount"]), 2),
        "total_transactions": int(row["total_transactions"]),
        "top_country": row.get("top_merchant_country", "n/c"),
        "top_category": row.get("top_merchant_category", "n/c"),
    }


def predict_segment(features: dict, kmeans, scaler, feature_columns, df_clustered) -> dict:
    """
    Predit le segment d'un nouvel utilisateur a partir de son profil comportemental.

    Les cles descriptives (pays, devise, categorie) sont ignorees si elles sont
    fournies : elles ne font pas partie du modele.
    """
    features = {k: v for k, v in features.items() if k not in DESCRIPTIVE_COLUMNS}
    df_input = pd.DataFrame([features])

    # Compression log - identique a features.py, appliquee AVANT le scaler
    for col in LOG_COLUMNS:
        if col in df_input.columns:
            df_input[col] = np.log1p(df_input[col])

    cat_presentes = [c for c in CAT_COLUMNS if c in df_input.columns]
    if cat_presentes:
        df_input = pd.get_dummies(df_input, columns=cat_presentes, drop_first=True)

    # Aligne sur les colonnes vues a l'entrainement (ordre compris).
    df_encoded = df_input.reindex(columns=feature_columns, fill_value=0)

    df_scaled = pd.DataFrame(scaler.transform(df_encoded), columns=feature_columns)
    cluster_id = int(kmeans.predict(df_scaled)[0])

    segment_map = dict(
        df_clustered[["cluster", "segment"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    segment = segment_map.get(cluster_id, f"Cluster {cluster_id}")

    return {
        "segment": segment,
        "cluster_id": cluster_id,
        "description": SEGMENT_DESCRIPTIONS.get(segment, ""),
        "action": SEGMENT_ACTIONS.get(segment, ""),
    }


def segments_summary(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """Resume par segment : nombre d'utilisateurs, depense moyenne, panier moyen."""
    return (
        df_clustered.groupby("segment")
        .agg(
            nb_utilisateurs=("user_id", "count"),
            depense_moyenne=("total_spent", "mean"),
            panier_moyen=("avg_transaction_amount", "mean"),
            nb_transactions_moyen=("total_transactions", "mean"),
        )
        .round(2)
        .reset_index()
    )


def category_options(df_clustered: pd.DataFrame, column: str) -> list[str]:
    """Valeurs possibles d'une colonne categorielle, pour les menus deroulants."""
    if column not in df_clustered.columns:
        return []
    return sorted(df_clustered[column].dropna().unique().tolist())