# Segmentation client — pipeline de bout en bout

> Segmentation comportementale des clients d'une néo-banque à partir de
> 2,7 M de transactions : de l'exploration au pipeline industrialisé, avec
> une API REST et une interface de consultation.

**Démo en ligne : https://pipeline-segmentation-client.streamlit.app/

---

## Le problème

Une néo-banque envoie les mêmes offres à tous ses clients. Sans segmentation,
un utilisateur qui paie 300 fois par mois chez 27 commerçants reçoit la même
communication qu'un compte qui ne sert qu'à recevoir des virements.

L'objectif : identifier automatiquement des groupes d'usage distincts, pour
que l'équipe marketing puisse cibler.

---

## Les segments obtenus

Trois profils se dégagent, ordonnés par **taux d'usage marchand** — la part
des transactions effectuées chez un commerçant.

| Segment | Utilisateurs | Achats marchands | Commerçants distincts | Transactions | Panier moyen |
|---|---|---|---|---|---|
| **Compte de réception** | 4 595 | 8 % | 0,4 | 9 | 138 |
| **Usage partiel** | 6 665 | 56 % | 8,3 | 39 | 80 |
| **Compte principal** | 7 269 | 62 % | 26,6 | 289 | 68 |

La lecture est nette. Le premier groupe **reçoit** de l'argent (52 % de
transactions entrantes) mais ne paie quasiment pas : c'est un compte
secondaire. Le troisième est le compte du quotidien. Le deuxième est
l'entre-deux, avec un usage marchand réel mais limité.

| Segment | Action recommandée |
|---|---|
| Compte de réception | Convertir en compte de paiement : carte physique, domiciliation de revenus |
| Usage partiel | Élargir l'usage : cashback ciblé sur les catégories encore payées ailleurs |
| Compte principal | Fidéliser : offre premium, ce sont les clients à retenir |

---

## Les données

- **2 740 075 transactions**, 18 766 utilisateurs, 12 colonnes
- Après filtrage sur les transactions abouties (`COMPLETED`) : **18 529 utilisateurs**
- 332 107 lignes écartées, dont 155 286 refus et 112 618 annulations —
  elles étaient auparavant comptées comme des dépenses réelles

---

## Ce que j'ai corrigé en cours de route

Cette section est volontairement détaillée : les trois problèmes ci-dessous
n'étaient pas visibles dans les métriques du modèle, seulement en relisant
les résultats.

### 1. Les transactions refusées comptaient comme des dépenses

La dépense moyenne du segment le plus élevé ressortait à 35 M — invraisemblable
pour une néo-banque. Aucun filtre n'existait sur `transactions_state`.
Après correction : 16 483.

### 2. Le mode calculé sur des valeurs manquantes

42 % des transactions n'ont pas de commerçant (virements, transferts). Ces
valeurs nulles étaient remplacées par une sentinelle `unknown_country` **avant**
le calcul du mode, qui l'élisait donc chez presque tous les clients. Tous les
profils affichaient « pays inconnu ». Corrigé avec un `FILTER` sur l'agrégat :

```sql
COALESCE(
    MODE(ea_merchant_country) FILTER (WHERE ea_merchant_country <> 'unknown_country'),
    'unknown_country'
) AS top_merchant_country
```

### 3. Le modèle segmentait par géographie, pas par comportement

C'est l'erreur la plus intéressante. Sur 175 features, 165 étaient des
colonnes one-hot de pays et de devise. En vérifiant la modalité dominante de
chaque cluster, le verdict était sans appel :

| Cluster | Pays dominant | Profil numérique |
|---|---|---|
| 0 | GBR | 162 transactions, panier 78 |
| 2 | FRA | 146 transactions, panier 69 |

Le modèle avait appris à séparer les Britanniques des Français. Deux causes
cumulées : les indicatrices binaires (0/1) écrasaient les variables continues,
elles-mêmes tassées autour de 0,003 après `MinMaxScaler` à cause de quelques
clients à plusieurs millions.

Deux corrections :

- **Compression log** (`np.log1p`) des montants et comptages avant le scaling.
  Les valeurs s'étalent désormais entre 0,3 et 0,8 au lieu de 1e-08.
- **Retrait du pays, de la devise et de la catégorie marchande** du modèle.
  Elles restent dans les données pour décrire les segments a posteriori, mais
  ce sont des attributs d'identité, pas des comportements.

On passe de 175 features à 9, toutes comportementales — et les segments
deviennent interprétables.

---

## Architecture

```
rev-transactions.csv (2,7 M lignes)
        │
        ▼
   ingest.py      agrégation par utilisateur (DuckDB)
        │         filtre COMPLETED, mode hors valeurs nulles
        ▼
   features.py    log1p + MinMaxScaler → 9 features comportementales
        │
        ▼
   train.py       KMeans (k=3), méthode du coude, nommage des segments
        │
        ├──────────────┐
        ▼              ▼
    core.py ────► api.py (FastAPI)   app.py (Streamlit)
```

`core.py` porte la logique métier partagée : chargement des artefacts,
prédiction, résumés. L'API et l'interface en sont deux couches de
présentation. Sans ce module, l'encodage et le scaling seraient dupliqués —
et divergeraient à la première évolution du modèle.

---

## Les 9 features

| Feature | Description |
|---|---|
| `total_spent` | Dépense totale (USD) |
| `avg_transaction_amount` | Panier moyen |
| `total_transactions` | Nombre d'opérations |
| `inbound_transactions` / `outbound_transactions` | Entrantes / sortantes |
| `unique_merchants` | Commerçants distincts (hors virements) |
| `merchant_transactions` | Opérations chez un commerçant |
| `inbound_ratio` | Part d'opérations entrantes |
| `merchant_ratio` | Part d'opérations marchandes |

Les deux ratios sont les plus discriminants : ce sont eux qui distinguent des
styles d'usage à volume comparable.

---

## Stack

| Composant | Rôle |
|---|---|
| DuckDB | Agrégation SQL de 2,7 M lignes sans charger en mémoire |
| Pandas | Manipulation des DataFrames |
| Scikit-learn | KMeans, MinMaxScaler |
| Streamlit | Interface de consultation |
| FastAPI + Uvicorn | Exposition du modèle en REST |
| Parquet | Format intermédiaire entre les étapes |

---

## Installation

```bash
git clone https://github.com/Mak-lab25/Pipeline-d-une-segmentation-client
cd Pipeline-d-une-segmentation-client
pip install -r requirements.txt
```

Le fichier source `rev-transactions.csv` (2,7 M lignes) n'est pas versionné.
Un extrait de 1 000 lignes est fourni dans `samples/` pour tester la chaîne.

## Lancer le pipeline

```bash
python ingest.py      # agrégation par utilisateur
python features.py    # transformation et scaling
python train.py       # entraînement KMeans
```

## Lancer l'interface

```bash
streamlit run app.py
```

Trois onglets : vue d'ensemble des segments, consultation d'un client
existant, et simulation d'un nouveau profil.

## Lancer l'API

```bash
uvicorn api:app --reload
```

Documentation interactive sur `http://localhost:8000/docs`.

| Endpoint | Description |
|---|---|
| `GET /segment?user_id=...` | Segment d'un utilisateur existant |
| `POST /predict` | Segment d'un nouveau profil |
| `GET /segments` | Résumé de tous les segments |

---

## Structure

```
├── ingest.py                 # étape 1 — agrégation DuckDB
├── features.py               # étape 2 — transformation et scaling
├── train.py                  # étape 3 — KMeans et nommage
├── core.py                   # logique partagée API / interface
├── api.py                    # API FastAPI
├── app.py                    # interface Streamlit
├── theme.py                  # identité visuelle de l'interface
├── .streamlit/config.toml    # thème Streamlit
├── app_data/                 # artefacts allégés (versionnés pour le déploiement)
├── data/                     # artefacts complets (non versionnés)
├── samples/                  # extrait de 1 000 transactions
├── LeWagon_clustering.ipynb  # exploration initiale (EDA + premier clustering)
└── Projet_clustering_wagon.pdf
```

---

## Limites connues et pistes

- **Les transactions annulées (`REVERTED`) sont écartées.** Un taux de
  reversement élevé est pourtant un signal comportemental intéressant : il
  mériterait sa propre feature.
- **Les variables sont fortement corrélées entre elles.** `total_spent` dépend
  mécaniquement de `total_transactions`. Le modèle capte donc surtout un axe
  d'intensité d'usage ; une ACP préalable ou un jeu de variables purement
  relatives affinerait la séparation.
- **La méthode du coude suggère k=4.** L'inertie chute encore nettement entre
  k=3 (2 401) et k=4 (2 058). Les trois segments actuels sont lisibles, mais un
  quatrième groupe existe probablement.
- **Les transformations sont dupliquées** entre `features.py` et `core.py`.
  Un `Pipeline` scikit-learn persisté (`ColumnTransformer` +
  `FunctionTransformer(np.log1p)` + `MinMaxScaler`) supprimerait ce risque
  de divergence.
- **Aucune validation de la stabilité des clusters.** Un score de silhouette
  et un test sur plusieurs graines confirmeraient la robustesse du k choisi.

---

## Cas d'usage similaires

Le pipeline s'adapte à toute segmentation comportementale : e-commerce,
scoring d'utilisateurs d'application mobile, abonnés d'un service SaaS,
analyse client en banque ou en assurance.
