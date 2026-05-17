# 🏦 Customer Segmentation Pipeline — NexBank

> **Segmentation automatique des clients d'une néo-banque** à partir de leurs habitudes de transaction, avec un pipeline de données modulaire et une API de prédiction en temps réel.

---

## 🎯 Le problème business

Une néo-banque comme NexBank génère des millions de transactions par jour. Sans segmentation, toutes les communications et offres sont identiques pour tous les clients — ce qui est inefficace.

**L'objectif :** identifier automatiquement des groupes d'utilisateurs aux comportements similaires pour permettre à l'équipe marketing de personnaliser ses actions.

| Segment | Profil | Action recommandée |
|---|---|---|
| 🟢 High Spender | Transactions fréquentes, montants élevés | Offres premium, programme fidélité |
| 🔵 Mid Spender | Usage régulier, montants modérés | Activation de nouveaux services |
| 🔴 Low Spender | Faible engagement | Campagnes de réactivation |

---

## 📊 Les données

- **2 740 075 transactions** financières
- **18 766 utilisateurs uniques**
- **12 colonnes** : type de transaction, montant, devise, direction, pays marchand, catégorie MCC...

---

## 🏗️ Architecture du pipeline

```
rev-transactions.csv (2.7M lignes)
        ↓
   ingest.py          → nettoyage + agrégation par utilisateur (DuckDB)
        ↓
   data/df_user.parquet
        ↓
   features.py        → encodage One-Hot + normalisation MinMaxScaler
        ↓
   data/df_scaled.parquet
        ↓
   train.py           → KMeans (k=3) + elbow method + nommage des segments
        ↓
   data/kmeans_model.pkl
        ↓
   api.py             → API FastAPI exposant le modèle via 3 endpoints
```

Chaque script a une responsabilité unique — si le modèle change, on retouche uniquement `train.py` sans toucher au reste.

---

## 🚀 Résultats

- **3 segments** identifiés via méthode du coude (elbow method)
- **18 766 utilisateurs** profilés sur 140 features
- **API REST** permettant d'interroger le segment d'un utilisateur en temps réel

---

## ⚙️ Stack technique

| Composant | Rôle |
|---|---|
| `DuckDB` | Lecture et agrégation SQL de 2.7M transactions sans surcharge mémoire |
| `Pandas` | Manipulation des DataFrames |
| `Scikit-learn` | KMeans, MinMaxScaler, PCA |
| `FastAPI` | Exposition du modèle via API REST |
| `Uvicorn` | Serveur ASGI pour lancer l'API |
| `Parquet` | Format de stockage intermédiaire entre les étapes |

---

## 🛠️ Installation

```bash
git clone https://github.com/ton-username/customer-segmentation
cd customer-segmentation
pip install -r requirements.txt
```

---

## ▶️ Lancer le pipeline

Les scripts s'exécutent dans l'ordre suivant :

```bash
# Étape 1 — Nettoyage et agrégation des transactions
python ingest.py

# Étape 2 — Encodage et normalisation
python features.py

# Étape 3 — Entraînement KMeans
python train.py

# Étape 4 — Lancer l'API
uvicorn api:app --reload
```

---

## 🔌 API — Endpoints

Une fois l'API lancée, ouvre `http://localhost:8000/docs` pour accéder à la documentation interactive.

### `GET /segment?user_id=user_898`
Retourne le segment d'un utilisateur existant.

```json
{
  "user_id": "user_898",
  "segment": "Low Spender",
  "description": "Utilisateur avec peu de transactions et des montants faibles.",
  "total_spent": 6507.06,
  "avg_transaction": 24.93,
  "total_transactions": 261
}
```

### `POST /predict`
Prédit le segment d'un **nouvel** utilisateur à partir de ses données.

```json
{
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
```

Réponse :
```json
{
  "segment": "Mid Spender",
  "cluster_id": 2,
  "description": "Utilisateur régulier avec des montants modérés."
}
```

### `GET /segments`
Retourne le résumé de tous les segments.

```json
[
  {"segment": "High Spender", "nb_utilisateurs": 8640, "total_spent_moyen": 35665099.9},
  {"segment": "Mid Spender",  "nb_utilisateurs": 6286, "total_spent_moyen": 17251256.15},
  {"segment": "Low Spender",  "nb_utilisateurs": 3840, "total_spent_moyen": 13049096.37}
]
```

---

## 📁 Structure du projet

```
customer-segmentation/
├── ingest.py              # Étape 1 : lecture CSV + agrégation DuckDB
├── features.py            # Étape 2 : encodage + scaling
├── train.py               # Étape 3 : KMeans + elbow method
├── api.py                 # Étape 4 : API FastAPI
├── requirements.txt
├── README.md
├── data/
│   ├── df_user.parquet        # Features par utilisateur (produit par ingest.py)
│   ├── df_scaled.parquet      # Features scalées (produit par features.py)
│   ├── df_clustered.parquet   # Résultats avec segments (produit par train.py)
│   ├── kmeans_model.pkl       # Modèle entraîné
│   ├── scaler.pkl             # Scaler pour nouvelles prédictions
│   ├── feature_columns.pkl    # Colonnes attendues par le modèle
│   ├── cluster_summary.csv    # Profil moyen par segment
│   └── elbow_plot.png         # Graphique méthode du coude
└── samples/
    └── rev-transactions-sample.csv   # Extrait anonymisé pour test
```

> ⚠️ Le fichier `rev-transactions.csv` (2.7M lignes) n'est pas inclus dans le repo pour des raisons de taille. Un extrait de 1000 lignes est disponible dans `samples/`.

---

## 🤝 Cas d'usage similaires

Ce pipeline est adaptable à tout contexte de segmentation comportementale :
- Segmentation clients e-commerce
- Scoring utilisateurs d'une app mobile
- Segmentation abonnés d'un service SaaS
- Analyse comportementale dans le secteur bancaire ou assurance
