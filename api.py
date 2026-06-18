"""
api.py
------
API FastAPI qui expose le modèle KMeans via un endpoint REST.
Permet de prédire le segment d'un utilisateur à partir de ses données.

Endpoints :
    GET  /                        → message de bienvenue
    GET  /segment?user_id=...     → segment d'un utilisateur existant
    POST /predict                 → segment pour un nouvel utilisateur

Lancer avec :
    uvicorn api:app --reload
"""

import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

MODEL_PATH      = "data/kmeans_model.pkl"
CLUSTERED_PATH  = "data/df_clustered.parquet"
SCALER_PATH     = "data/scaler.pkl"
COLUMNS_PATH    = "data/feature_columns.pkl"

SEGMENT_DESCRIPTIONS = {
    "Low Spender":  "Utilisateur avec peu de transactions et des montants faibles.",
    "Mid Spender":  "Utilisateur régulier avec des montants modérés.",
    "High Spender": "Utilisateur très actif avec des montants élevés.",
}

# =========================
# CHARGEMENT DES RESSOURCES
# =========================

def load_resources():
    """Charge le modèle, le scaler et les données au démarrage de l'API."""

    print("⏳ Chargement des ressources...")

    # Modèle KMeans
    with open(MODEL_PATH, "rb") as f:
        kmeans = pickle.load(f)

    # Scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Colonnes attendues par le modèle
    with open(COLUMNS_PATH, "rb") as f:
        feature_columns = pickle.load(f)

    # Résultats complets (user_id → segment)
    df_clustered = pd.read_parquet(CLUSTERED_PATH)

    print("✅ Ressources chargées")
    return kmeans, scaler, feature_columns, df_clustered


# Chargement au démarrage
kmeans, scaler, feature_columns, df_clustered = load_resources()


# =========================
# INITIALISATION FASTAPI
# =========================

app = FastAPI(
    title="Customer Segmentation API",
    description="Prédit le segment client (Low / Mid / High Spender) à partir des données de transactions.",
    version="1.0.0"
)


# =========================
# SCHÉMA DE DONNÉES (POST /predict)
# =========================

class UserFeatures(BaseModel):
    """
    Données d'un nouvel utilisateur pour prédire son segment.
    Correspond aux features construites dans ingest.py.
    """
    total_spent:            float
    avg_transaction_amount: float
    total_transactions:     int
    inbound_transactions:   int
    outbound_transactions:  int
    unique_merchants:       int
    inbound_ratio:          float
    top_merchant_country:   str = "GB"
    top_merchant_currency:  str = "GBP"
    top_merchant_category:  str = "Unknown"

    class Config:
        json_schema_extra = {
            "example": {
                "total_spent": 15000.0,
                "avg_transaction_amount": 250.0,
                "total_transactions": 60,
                "inbound_transactions": 10,
                "outbound_transactions": 50,
                "unique_merchants": 8,
                "inbound_ratio": 0.17,
                "top_merchant_country": "GB",
                "top_merchant_currency": "GBP",
                "top_merchant_category": "Miscellaneous stores"
            }
        }


# =========================
# ENDPOINTS
# =========================

@app.get("/")
def root():
    """Message de bienvenue."""
    return {
        "message": "Customer Segmentation API",
        "endpoints": {
            "GET  /segment?user_id=user_898": "Segment d'un utilisateur existant",
            "POST /predict":                  "Segment pour un nouvel utilisateur",
            "GET  /segments":                 "Résumé de tous les segments",
            "GET  /docs":                     "Documentation interactive"
        }
    }


@app.get("/segment")
def get_segment(user_id: str):
    """
    Retourne le segment d'un utilisateur existant dans les données.

    Exemple : GET /segment?user_id=user_898
    """
    # Chercher l'utilisateur dans les données
    user_row = df_clustered[df_clustered["user_id"] == user_id]

    if user_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Utilisateur '{user_id}' introuvable."
        )

    row     = user_row.iloc[0]
    segment = row["segment"]

    return {
        "user_id":               user_id,
        "segment":               segment,
        "description":           SEGMENT_DESCRIPTIONS.get(segment, ""),
        "total_spent":           round(float(row["total_spent"]), 2),
        "avg_transaction":       round(float(row["avg_transaction_amount"]), 2),
        "total_transactions":    int(row["total_transactions"]),
        "top_country":           row["top_merchant_country"],
        "top_category":          row["top_merchant_category"],
    }


@app.post("/predict")
def predict_segment(user: UserFeatures):
    """
    Prédit le segment d'un nouvel utilisateur à partir de ses features.

    Le corps de la requête doit contenir les données de l'utilisateur.
    """
    # Construire un DataFrame avec les données reçues
    data = {
        "total_spent":            [user.total_spent],
        "avg_transaction_amount": [user.avg_transaction_amount],
        "total_transactions":     [user.total_transactions],
        "inbound_transactions":   [user.inbound_transactions],
        "outbound_transactions":  [user.outbound_transactions],
        "unique_merchants":       [user.unique_merchants],
        "inbound_ratio":          [user.inbound_ratio],
        "top_merchant_country":   [user.top_merchant_country],
        "top_merchant_currency":  [user.top_merchant_currency],
        "top_merchant_category":  [user.top_merchant_category],
    }
    df_input = pd.DataFrame(data)

    # One-hot encoding — même logique que features.py
    df_encoded = pd.get_dummies(
        df_input,
        columns=["top_merchant_category", "top_merchant_country", "top_merchant_currency"],
        drop_first=True
    )

    # Aligner les colonnes avec celles du modèle
    # (ajouter les colonnes manquantes avec 0, supprimer les inconnues)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scaler avec le même scaler que l'entraînement
    df_scaled = pd.DataFrame(
        scaler.transform(df_encoded),
        columns=feature_columns
    )

    # Prédiction
    cluster_id = int(kmeans.predict(df_scaled)[0])

    # Retrouver le nom du segment depuis les données existantes
    segment_map = df_clustered[["cluster", "segment"]].drop_duplicates()
    segment_map = dict(zip(segment_map["cluster"], segment_map["segment"]))
    segment = segment_map.get(cluster_id, f"Cluster {cluster_id}")

    return {
        "segment":     segment,
        "cluster_id":  cluster_id,
        "description": SEGMENT_DESCRIPTIONS.get(segment, ""),
        "input":       user.model_dump()
    }


@app.get("/segments")
def get_all_segments():
    """
    Retourne le résumé de tous les segments :
    nombre d'utilisateurs, dépense moyenne, panier moyen.
    """
    summary = df_clustered.groupby("segment").agg(
        nb_utilisateurs=("user_id", "count"),
        total_spent_moyen=("total_spent", "mean"),
        panier_moyen=("avg_transaction_amount", "mean"),
        nb_transactions_moyen=("total_transactions", "mean")
    ).round(2).reset_index()

    return summary.to_dict(orient="records")
