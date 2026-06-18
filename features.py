"""
features.py
-----------
Préparation des features pour le clustering KMeans :
- Encodage One-Hot des variables catégorielles
- Normalisation MinMaxScaler
- Sauvegarde du DataFrame scalé + du scaler

Lit  : data/df_user.parquet    (produit par ingest.py)
Écrit: data/df_scaled.parquet  (pour train.py)
       data/scaler.pkl          (pour api.py)
       data/feature_columns.pkl (pour api.py)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

INPUT_PATH         = "data/df_user.parquet"
OUTPUT_SCALED      = "data/df_scaled.parquet"
OUTPUT_SCALER      = "data/scaler.pkl"
OUTPUT_COLUMNS     = "data/feature_columns.pkl"

# Colonnes catégorielles à encoder (same as notebook cellule 41)
CATEGORICAL_COLS = [
    "top_merchant_category",
    "top_merchant_country",
    "top_merchant_currency"
]


# =========================
# CHARGEMENT
# =========================

def load_user_features(input_path: str = INPUT_PATH) -> pd.DataFrame:
    """Lit le fichier parquet produit par ingest.py."""
    print(f"📂 Lecture de {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"✅ {len(df):,} utilisateurs chargés — {len(df.columns)} colonnes")
    return df


# =========================
# ENCODAGE ONE-HOT
# =========================

def encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Supprime user_id (pas une feature pour le modèle)
    - Applique get_dummies sur les colonnes catégorielles
    Reproduit la cellule 41 du notebook.
    """
    print("🔠 Encodage One-Hot des variables catégorielles...")

    # On garde user_id de côté pour pouvoir le réassocier après
    user_ids = df["user_id"].copy()

    # Supprimer user_id avant encodage
    df = df.drop(columns=["user_id"])

    # One-hot encoding — même chose que dans le notebook
    df_encoded = pd.get_dummies(
        df,
        columns=CATEGORICAL_COLS,
        drop_first=True       # évite la multicolinéarité
    )

    # S'assurer que tout est numérique
    df_encoded = df_encoded.select_dtypes(include=["float64", "int64", "bool", "uint8"])

    # Remplir les éventuels NaN restants par 0
    df_encoded = df_encoded.fillna(0)

    print(f"✅ {len(df_encoded.columns)} features après encodage")
    return df_encoded, user_ids


# =========================
# SCALING
# =========================

def scale(df_encoded: pd.DataFrame):
    """
    Normalise toutes les features entre 0 et 1 avec MinMaxScaler.
    Reproduit la cellule 44 du notebook.
    Retourne le DataFrame scalé ET le scaler (pour l'API).
    """
    print("⚖️  Normalisation MinMaxScaler...")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_encoded)

    df_scaled = pd.DataFrame(
        scaled_values,
        columns=df_encoded.columns
    )

    print(f"✅ Scaling terminé — valeurs entre 0 et 1")
    return df_scaled, scaler


# =========================
# SAUVEGARDE
# =========================

def save_outputs(
    df_scaled: pd.DataFrame,
    user_ids: pd.Series,
    scaler: MinMaxScaler,
    output_scaled: str  = OUTPUT_SCALED,
    output_scaler: str  = OUTPUT_SCALER,
    output_columns: str = OUTPUT_COLUMNS
):
    """Sauvegarde les fichiers nécessaires aux étapes suivantes."""

    Path(output_scaled).parent.mkdir(parents=True, exist_ok=True)

    # Ajouter user_id au DataFrame scalé pour pouvoir retrouver les utilisateurs
    df_scaled_with_id = df_scaled.copy()
    df_scaled_with_id.insert(0, "user_id", user_ids.values)

    # Parquet pour train.py
    df_scaled_with_id.to_parquet(output_scaled, index=False)
    print(f"💾 DataFrame scalé sauvegardé → {output_scaled}")

    # Scaler pour api.py (on en aura besoin pour scaler de nouvelles données)
    with open(output_scaler, "wb") as f:
        pickle.dump(scaler, f)
    print(f"💾 Scaler sauvegardé → {output_scaler}")

    # Noms des colonnes pour api.py
    feature_columns = df_scaled.columns.tolist()
    with open(output_columns, "wb") as f:
        pickle.dump(feature_columns, f)
    print(f"💾 Colonnes sauvegardées → {output_columns}")


# =========================
# PIPELINE PRINCIPAL
# =========================

def run(input_path: str = INPUT_PATH):
    """Lance le pipeline complet : load → encode → scale → save."""

    # 1. Charger les features utilisateur
    df_user = load_user_features(input_path)

    # 2. Encoder les variables catégorielles
    df_encoded, user_ids = encode(df_user)

    # 3. Scaler
    df_scaled, scaler = scale(df_encoded)

    # 4. Sauvegarder
    save_outputs(df_scaled, user_ids, scaler)

    print(f"\n✅ features.py terminé — {len(df_scaled.columns)} features prêtes pour KMeans")
    print(f"📋 Colonnes : {df_scaled.columns.tolist()[:5]}... (et {len(df_scaled.columns)-5} autres)")

    return df_scaled


# =========================
# POINT D'ENTRÉE
# =========================

if __name__ == "__main__":
    df_scaled = run()
    print("\n📊 Aperçu :")
    print(df_scaled.head())
    print("\n📐 Shape :", df_scaled.shape)
