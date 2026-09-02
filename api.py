"""
api.py
------
API FastAPI exposant le modele de segmentation.

Couche HTTP uniquement : toute la logique metier (chargement des artefacts,
transformations, prediction) vit dans core.py, partagee avec l'interface
Streamlit. Ce decoupage garantit que l'API et l'app renvoient le meme
segment pour un meme profil.

Lancer avec :
    uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import core

# =========================
# CHARGEMENT AU DEMARRAGE
# =========================

print("Chargement des ressources...")
kmeans, scaler, feature_columns, df_clustered = core.load_resources()
print(f"Ressources chargees - {len(df_clustered):,} utilisateurs, "
      f"{len(feature_columns)} features")


app = FastAPI(
    title="Customer Segmentation API",
    description=(
        "Segmente les clients d'une neo-banque selon leur comportement "
        "d'usage : compte de reception, usage partiel ou compte principal."
    ),
    version="2.0.0",
)


# =========================
# SCHEMA D'ENTREE
# =========================

class UserFeatures(BaseModel):
    """
    Profil comportemental d'un utilisateur.

    Correspond aux features construites par ingest.py. Les attributs
    descriptifs (pays, devise, categorie marchande) ne figurent pas ici :
    ils ont ete retires du modele, qui segmente sur le comportement seul.
    """

    total_spent: float = Field(ge=0, description="Depense totale en USD")
    avg_transaction_amount: float = Field(ge=0, description="Panier moyen")
    total_transactions: int = Field(ge=0)
    inbound_transactions: int = Field(ge=0)
    outbound_transactions: int = Field(ge=0)
    unique_merchants: int = Field(ge=0, description="Commercants distincts")
    merchant_transactions: int = Field(ge=0, description="Operations chez un commercant")
    inbound_ratio: float = Field(ge=0, le=1)
    merchant_ratio: float = Field(ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_spent": 16483.0,
                "avg_transaction_amount": 68.0,
                "total_transactions": 289,
                "inbound_transactions": 49,
                "outbound_transactions": 240,
                "unique_merchants": 27,
                "merchant_transactions": 179,
                "inbound_ratio": 0.17,
                "merchant_ratio": 0.62,
            }
        }
    }


# =========================
# ENDPOINTS
# =========================

@app.get("/")
def root():
    """Point d'entree : liste des endpoints disponibles."""
    return {
        "message": "Customer Segmentation API",
        "endpoints": {
            "GET /segment?user_id=user_898": "Segment d'un utilisateur existant",
            "POST /predict": "Segment d'un nouveau profil",
            "GET /segments": "Resume de tous les segments",
            "GET /docs": "Documentation interactive",
        },
    }


@app.get("/segment")
def get_segment(user_id: str):
    """Retourne le segment d'un utilisateur present dans les donnees."""
    infos = core.get_user_segment(df_clustered, user_id)
    if infos is None:
        raise HTTPException(
            status_code=404,
            detail=f"Utilisateur '{user_id}' introuvable.",
        )
    infos["action"] = core.SEGMENT_ACTIONS.get(infos["segment"], "")
    return infos


@app.post("/predict")
def predict(user: UserFeatures):
    """Predit le segment d'un nouveau profil comportemental."""
    resultat = core.predict_segment(
        user.model_dump(), kmeans, scaler, feature_columns, df_clustered
    )
    resultat["input"] = user.model_dump()
    return resultat


@app.get("/segments")
def get_all_segments():
    """Resume par segment : effectif, depense moyenne, panier moyen."""
    summary = core.segments_summary(df_clustered)
    records = summary.to_dict(orient="records")
    for r in records:
        r["description"] = core.SEGMENT_DESCRIPTIONS.get(r["segment"], "")
        r["action"] = core.SEGMENT_ACTIONS.get(r["segment"], "")
    return records