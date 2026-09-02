"""
train.py
--------
Entraînement du modèle KMeans sur les features scalées.
- Méthode du coude pour choisir le bon nombre de clusters
- Entraînement KMeans
- Sauvegarde du modèle et des résultats

Lit  : data/df_scaled.parquet   (produit par features.py)
Écrit: data/kmeans_model.pkl    (pour api.py)
       data/df_clustered.parquet (résultats complets)
       data/cluster_summary.csv  (profil moyen par segment)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

INPUT_PATH       = "data/df_scaled.parquet"
OUTPUT_MODEL     = "data/kmeans_model.pkl"
OUTPUT_CLUSTERED = "data/df_clustered.parquet"
OUTPUT_SUMMARY   = "data/cluster_summary.csv"
OUTPUT_ELBOW     = "data/elbow_plot.png"

N_CLUSTERS = 3     # nombre de clusters choisi (confirmé par elbow method)
K_RANGE    = range(1, 10)


# =========================
# CHARGEMENT
# =========================

def load_scaled_features(input_path: str = INPUT_PATH):
    """Lit le parquet produit par features.py."""
    print(f"📂 Lecture de {input_path}...")
    df = pd.read_parquet(input_path)

    # Séparer user_id des features
    user_ids = df["user_id"].copy()
    df_features = df.drop(columns=["user_id"])

    print(f"✅ {len(df):,} utilisateurs — {len(df_features.columns)} features")
    return df_features, user_ids


# =========================
# MÉTHODE DU COUDE
# =========================

def elbow_method(df_features: pd.DataFrame, k_range=K_RANGE):
    """
    Calcule l'inertie pour chaque valeur de k et sauvegarde le graphique.
    Reproduit la cellule 47 du notebook.
    """
    print("📐 Méthode du coude en cours...")

    inertia = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=300)
        kmeans.fit(df_features)
        inertia.append(kmeans.inertia_)
        print(f"   k={k} → inertia={kmeans.inertia_:,.0f}")

    # Sauvegarde du graphique
    plt.figure(figsize=(8, 5))
    plt.plot(list(k_range), inertia, marker="o", linestyle="-", color="#2b69db")
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Méthode du coude — Choix du nombre de clusters")
    plt.grid(True)
    Path(OUTPUT_ELBOW).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_ELBOW, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Graphique du coude sauvegardé → {OUTPUT_ELBOW}")

    return inertia


# =========================
# ENTRAÎNEMENT KMEANS
# =========================

def train_kmeans(df_features: pd.DataFrame, n_clusters: int = N_CLUSTERS):
    """
    Entraîne KMeans et retourne le modèle + les labels de clusters.
    Reproduit la cellule 49 du notebook.
    """
    print(f"\n🤖 Entraînement KMeans avec k={n_clusters}...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto",
        max_iter=300
    )
    clusters = kmeans.fit_predict(df_features)

    # Répartition des utilisateurs par cluster
    unique, counts = np.unique(clusters, return_counts=True)
    print("✅ Répartition par cluster :")
    for cluster_id, count in zip(unique, counts):
        print(f"   Cluster {cluster_id} → {count:,} utilisateurs ({count/len(clusters)*100:.1f}%)")

    return kmeans, clusters


# =========================
# NOMMAGE DES SEGMENTS
# =========================

def name_segments(df_clustered: pd.DataFrame, n_clusters: int) -> dict:
    """
    Attribue un nom métier à chaque cluster selon son taux d'usage marchand.

    Le tri se fait sur merchant_ratio (part des transactions effectuées chez
    un commerçant) plutôt que sur total_spent : c'est cette variable qui
    sépare réellement les clusters. Trier sur la dépense produisait des noms
    faux — le groupe le moins dépensier avait le panier moyen le plus élevé.

    Ne fonctionne que pour k=3 ; au-delà, retombe sur un libellé générique.
    """
    labels = ["Compte de réception", "Usage partiel", "Compte principal"]

    if n_clusters != len(labels):
        return {c: f"Segment {c}" for c in sorted(df_clustered["cluster"].unique())}

    means = df_clustered.groupby("cluster")["merchant_ratio"].mean().sort_values()
    return {cluster_id: labels[i] for i, cluster_id in enumerate(means.index)}


# =========================
# SAUVEGARDE
# =========================

def save_outputs(
    kmeans,
    df_features: pd.DataFrame,
    user_ids: pd.Series,
    clusters: np.ndarray,
):
    """Sauvegarde le modèle, les résultats et le résumé par segment."""

    Path(OUTPUT_MODEL).parent.mkdir(parents=True, exist_ok=True)

    # 1. Sauvegarder le modèle KMeans
    with open(OUTPUT_MODEL, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"\n💾 Modèle KMeans sauvegardé → {OUTPUT_MODEL}")

    # 2. Construire le DataFrame avec user_id + cluster
    # On repart des features NON scalées pour le résumé lisible
    df_result = pd.read_parquet("data/df_user.parquet")
    df_result["cluster"] = clusters

    # Nommer les segments
    segment_names = name_segments(df_result, len(np.unique(clusters)))
    df_result["segment"] = df_result["cluster"].map(segment_names)

    # Sauvegarder
    df_result.to_parquet(OUTPUT_CLUSTERED, index=False)
    print(f"💾 Résultats complets sauvegardés → {OUTPUT_CLUSTERED}")

    # 3. Résumé par segment (profil moyen — cellule 50 du notebook)
    numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "cluster"]

    summary = df_result.groupby("segment")[numeric_cols].mean().round(2)
    summary["nb_utilisateurs"] = df_result.groupby("segment")["user_id"].count()
    summary.to_csv(OUTPUT_SUMMARY)
    print(f"💾 Résumé par segment sauvegardé → {OUTPUT_SUMMARY}")

    return df_result, summary


# =========================
# PIPELINE PRINCIPAL
# =========================

def run(input_path: str = INPUT_PATH, n_clusters: int = N_CLUSTERS):
    """Lance le pipeline complet : load → elbow → train → save."""

    # 1. Charger les features scalées
    df_features, user_ids = load_scaled_features(input_path)

    # 2. Méthode du coude
    elbow_method(df_features)

    # 3. Entraîner KMeans
    kmeans, clusters = train_kmeans(df_features, n_clusters)

    # 4. Sauvegarder
    df_result, summary = save_outputs(kmeans, df_features, user_ids, clusters)

    print("\n📊 Profil moyen par segment :")
    print(summary[["total_spent", "avg_transaction_amount",
                   "total_transactions", "nb_utilisateurs"]].to_string())

    print(f"\n✅ train.py terminé — modèle prêt dans {OUTPUT_MODEL}")
    return kmeans, df_result


# =========================
# POINT D'ENTRÉE
# =========================

if __name__ == "__main__":
    kmeans, df_result = run()
