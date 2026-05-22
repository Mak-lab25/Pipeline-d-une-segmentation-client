# 🏦 Customer Segmentation Pipeline — NexBank

> Segmentation automatique des clients d'une néo-banque à partir de leurs habitudes de transaction — de l'analyse exploratoire aux recommandations business, jusqu'à l'industrialisation en pipeline de données et API REST.

---

## 🎯 Le problème business

Une néo-banque comme NexBank génère des millions de transactions par jour. Sans segmentation, toutes les communications et offres sont identiques pour tous les clients — ce qui est à la fois inefficace et coûteux.

**L'objectif :** identifier automatiquement des groupes d'utilisateurs aux comportements similaires, pour permettre à l'équipe marketing de personnaliser ses actions.

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

## 🔍 Démarche complète

Ce projet couvre l'intégralité du cycle data — de l'exploration à la mise en production.

### 1. Analyse exploratoire (EDA)

📓 [`notebook/LeWagon_clustering.ipynb`](notebook/LeWagon_clustering.ipynb)

- Distribution des transactions par type, devise, direction et catégorie marchande
- Analyse de la présence physique vs en ligne (`ea_cardholderpresence`)
- Test du Khi-2 pour identifier les associations significatives entre variables
- Catégorisation des codes MCC en familles lisibles (General services, Transportation, Retail...)
- Visualisations : barres empilées, boxplots, heatmaps, camemberts par devise

**Principales observations :**
- Les paiements par carte (`CARD_PAYMENT`) dominent largement les transactions sortantes
- EUR, CHF et GBP concentrent les montants les plus élevés
- Nombre élevé de refus de transactions en présentiel — piste d'amélioration UX identifiée

---

### 2. Segmentation client (Clustering)

📓 [`notebook/LeWagon_clustering.ipynb`](notebook/LeWagon_clustering.ipynb)

- Construction des features par utilisateur : `total_spent`, `avg_transaction_amount`, `inbound_ratio`, `top_merchant_category`, `top_merchant_country`...
- Nettoyage des valeurs manquantes (`fillna`, `dropna`)
- One-Hot encoding des variables catégorielles
- Normalisation MinMaxScaler
- Méthode du coude (elbow method) → **k=3 clusters optimal**
- Visualisation PCA pour confirmer la séparation des clusters

**Résultats :**

| Cluster | Profil | Nb utilisateurs |
|---|---|---|
| Cluster 0 | Gamme moyenne | 3 748 |
| Cluster 1 | Fort et moyen potentiel | 1 964 |
| Cluster 2 | Majorité à fort potentiel | 3 895 |

---

### 3. Recommandations business

📄 [`presentation/NexBank_clustering.pdf`](presentation/NexBank_clustering.pdf)

- **Personnalisation des offres** : avantages spécifiques aux clients à fort potentiel, promotions adaptées aux Mid Spenders
- **Réactivation** : campagnes ciblées pour encourager les utilisateurs à faible engagement
- **Optimisation UX** : améliorer la catégorisation des transactions et analyser les causes des refus en présentiel
- **Perspectives** : segmentation dynamique (temporelle, hiérarchique), affinage avec silhouette score et DBSCAN

---

### 4. Industrialisation — Pipeline engineering

Après la phase analytique, le projet a été restructuré en pipeline modulaire pour être réutilisable, maintenable et exposable via API.

```
rev-transactions.csv (2.7M lignes)
        ↓
   ingest.py      → nettoyage + agrégation par utilisateur (DuckDB)
        ↓
   features.py    → encodage One-Hot + normalisation MinMaxScaler
        ↓
   train.py       → KMeans (k=3) + elbow method + nommage des segments
        ↓
   api.py         → API FastAPI — 3 endpoints publics
```

Chaque script a une responsabilité unique — si le modèle change, on retouche uniquement `train.py` sans toucher au reste.

---

## 🚀 Résultats

- **3 segments** identifiés et nommés (Low / Mid / High Spender)
- **18 766 utilisateurs** profilés sur 140 features
- **API REST** permettant d'interroger le segment d'un utilisateur en temps réel
- Pipeline reproductible en 4 commandes

---

## ⚙️ Stack technique

| Composant | Rôle |
|---|---|
| `DuckDB` | Lecture et agrégation SQL de 2.7M transactions sans surcharge mémoire |
| `Pandas` | Manipulation des DataFrames |
| `Scikit-learn` | KMeans, MinMaxScaler, PCA |
| `Matplotlib / Seaborn` | Visualisations EDA |
| `FastAPI` | Exposition du modèle via API REST |
| `Uvicorn` | Serveur ASGI pour lancer l'API |
| `Parquet` | Format de stockage intermédiaire entre les étapes |

---

## 🛠️ Installation

```bash
git clone https://github.com/Mak-lab25/Pipeline-d-une-segmentation-client
cd Pipeline-d-une-segmentation-client
pip install -r requirements.txt
```

---

## ▶️ Lancer le pipeline

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

Puis ouvre `http://localhost:8000/docs` pour accéder à la documentation interactive de l'API.

---

## 🔌 API — Endpoints

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
Prédit le segment d'un nouvel utilisateur à partir de ses données.

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

### `GET /segments`
Retourne le résumé de tous les segments (nb utilisateurs, dépense moyenne, panier moyen).

---

## 📁 Structure du projet

```
Pipeline-d-une-segmentation-client/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── LeWagon_clustering.ipynb     # Analyse complète : EDA + clustering
├── presentation/
│   └── NexBank_clustering.pdf       # Résultats et recommandations business
├── ingest.py                        # Étape 1 : lecture CSV + agrégation DuckDB
├── features.py                      # Étape 2 : encodage + scaling
├── train.py                         # Étape 3 : KMeans + elbow method
├── api.py                           # Étape 4 : API FastAPI
└── samples/
    └── rev-transactions-sample.csv  # Extrait anonymisé (1000 lignes) pour test
```

> ⚠️ Le fichier `rev-transactions.csv` (2.7M lignes) n'est pas inclus pour des raisons de taille. Un extrait de 1000 lignes est disponible dans `samples/`.

---

## 🤝 Cas d'usage similaires

Ce pipeline est adaptable à tout contexte de segmentation comportementale :
- Segmentation clients e-commerce
- Scoring utilisateurs d'une app mobile
- Segmentation abonnés d'un service SaaS
- Analyse comportementale en secteur bancaire ou assurance
