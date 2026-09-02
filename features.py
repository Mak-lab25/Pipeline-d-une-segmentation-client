"""
features.py
-----------
Preparation des features pour le clustering KMeans.

Choix de modelisation : la segmentation porte uniquement sur le COMPORTEMENT
d'usage (montants, frequence, diversite, ratios). Pays, devise et categorie
marchande sont conserves dans df_user pour decrire les segments a posteriori,
mais retires du modele : ce sont des attributs geographiques ou sectoriels,
et leurs colonnes one-hot (0/1) ecrasaient les variables continues.

Lit  : data/df_user.parquet    (produit par ingest.py)
Ecrit: data/df_scaled.parquet  (pour train.py)
       data/scaler.pkl          (pour api.py et app.py)
       data/feature_columns.pkl (pour api.py et app.py)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =========================
# CONFIGURATION
# =========================

INPUT_PATH     = "data/df_user.parquet"
OUTPUT_SCALED  = "data/df_scaled.parquet"
OUTPUT_SCALER  = "data/scaler.pkl"
OUTPUT_COLUMNS = "data/feature_columns.pkl"

# Aucune variable categorielle dans le modele.
CATEGORICAL_COLS = []

# Conservees dans df_user pour decrire les segments, exclues du modele.
DESCRIPTIVE_COLS = [
    "top_merchant_country",
    "top_merchant_currency",
    "top_merchant_category",
]

# Colonnes de comptage / montant : distributions tres asymetriques,
# compressees en log avant scaling.
LOG_COLS = [
    "total_spent",
    "avg_transaction_amount",
    "total_transactions",
    "inbound_transactions",
    "outbound_transactions",
    "unique_merchants",
    "merchant_transactions",
]

# Ratios deja bornes entre 0 et 1 : ni log, ni transformation.
RATIO_COLS = ["inbound_ratio", "merchant_ratio"]


# =========================
# CHARGEMENT
# =========================

def load_user_features(input_path: str = INPUT_PATH) -> pd.DataFrame:
    """Lit le fichier parquet produit par ingest.py."""
    print(f"Lecture de {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"OK - {len(df):,} utilisateurs charges, {len(df.columns)} colonnes")
    return df


# =========================
# SELECTION & ENCODAGE
# =========================

def encode(df: pd.DataFrame):
    """
    - Met user_id de cote (identifiant, pas une feature)
    - Retire les colonnes descriptives non modelisees
    - Encode les eventuelles colonnes categorielles restantes
    """
    print("Selection des features comportementales...")

    user_ids = df["user_id"].copy()
    df = df.drop(columns=["user_id"])

    a_retirer = [c for c in DESCRIPTIVE_COLS if c in df.columns]
    if a_retirer:
        df = df.drop(columns=a_retirer)
        print(f"   retirees du modele : {', '.join(a_retirer)}")

    cat_presentes = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cat_presentes:
        df = pd.get_dummies(df, columns=cat_presentes, drop_first=True)
        print(f"   encodees en one-hot : {', '.join(cat_presentes)}")

    df_encoded = df.select_dtypes(include=["number", "bool"])
    df_encoded = df_encoded.fillna(0)

    manquantes = [c for c in RATIO_COLS if c not in df_encoded.columns]
    if manquantes:
        print(f"   ATTENTION - ratios absents de df_user : {', '.join(manquantes)}")

    print(f"OK - {len(df_encoded.columns)} features retenues")
    return df_encoded, user_ids


# =========================
# SCALING
# =========================

def scale(df_encoded: pd.DataFrame):
    """
    Compresse les comptages et montants en log, puis normalise entre 0 et 1.
    Retourne le DataFrame scale ET le scaler (reutilise a la prediction).
    """
    print("Compression log des montants, puis MinMaxScaler...")

    df_encoded = df_encoded.copy()
    for col in LOG_COLS:
        if col in df_encoded.columns:
            # log1p gere le zero sans erreur.
            df_encoded[col] = np.log1p(df_encoded[col])

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_encoded)
    df_scaled = pd.DataFrame(scaled_values, columns=df_encoded.columns)

    print("OK - valeurs entre 0 et 1")
    return df_scaled, scaler


# =========================
# SAUVEGARDE
# =========================

def save_outputs(
    df_scaled: pd.DataFrame,
    user_ids: pd.Series,
    scaler: MinMaxScaler,
    output_scaled: str = OUTPUT_SCALED,
    output_scaler: str = OUTPUT_SCALER,
    output_columns: str = OUTPUT_COLUMNS,
):
    """Sauvegarde les fichiers necessaires aux etapes suivantes."""
    Path(output_scaled).parent.mkdir(parents=True, exist_ok=True)

    # user_id reinsere pour permettre a train.py de joindre sur la cle
    # plutot que sur l'ordre des lignes.
    df_scaled_with_id = df_scaled.copy()
    df_scaled_with_id.insert(0, "user_id", user_ids.values)

    df_scaled_with_id.to_parquet(output_scaled, index=False)
    print(f"Sauvegarde -> {output_scaled}")

    with open(output_scaler, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Sauvegarde -> {output_scaler}")

    feature_columns = df_scaled.columns.tolist()
    with open(output_columns, "wb") as f:
        pickle.dump(feature_columns, f)
    print(f"Sauvegarde -> {output_columns}")


# =========================
# PIPELINE PRINCIPAL
# =========================

def run(input_path: str = INPUT_PATH):
    """Lance le pipeline complet : load -> encode -> scale -> save."""
    df_user = load_user_features(input_path)
    df_encoded, user_ids = encode(df_user)
    df_scaled, scaler = scale(df_encoded)
    save_outputs(df_scaled, user_ids, scaler)

    print(f"\nfeatures.py termine - {len(df_scaled.columns)} features pretes pour KMeans")
    print(f"Colonnes : {df_scaled.columns.tolist()}")
    return df_scaled


if __name__ == "__main__":
    df_scaled = run()
    print("\nApercu :")
    print(df_scaled.head())
    print("\nShape :", df_scaled.shape)
    print("\nStatistiques (verifier que les features s'etalent bien sur [0,1]) :")
    print(df_scaled.describe().round(3).T[["mean", "std", "min", "max"]])