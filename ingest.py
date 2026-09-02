"""
ingest.py
---------
Chargement et préparation des données de transactions
en utilisant DuckDB pour le traitement à grande échelle.

Remplace les cellules 0-38 du notebook LeWagon_clustering.ipynb.
"""

import duckdb
import pandas as pd
from pathlib import Path

# =========================
# CONFIGURATION
# =========================

CSV_PATH    = "rev-transactions.csv"
OUTPUT_PATH = "data/df_user.parquet"   # fichier intermédiaire pour features.py


# =========================
# MAPPING MCC → CATÉGORIE
# =========================
# Reproduit fidèlement la fonction category_mcc() du notebook

MCC_CATEGORIES = [
    ((1500, 2999), "General services"),
    ((3000, 3299), "Airlines"),
    ((3500, 3999), "Lodging"),
    ((4000, 4799), "Transportation services"),
    ((4800, 4999), "Utility services"),
    ((5000, 5599), "Retail outlet services"),
    ((5600, 5699), "Clothing stores"),
    ((5700, 7299), "Miscellaneous stores"),
    ((7300, 7999), "Business services"),
    ((8000, 8999), "Professional services"),
]

def build_mcc_case() -> str:
    """Construit une expression SQL CASE pour mapper MCC → catégorie."""
    cases = "\n        ".join([
        f"WHEN ea_merchant_mcc BETWEEN {low} AND {high} THEN '{label}'"
        for (low, high), label in MCC_CATEGORIES
    ])
    return f"""
        CASE
        {cases}
        ELSE 'Unknown'
        END
    """


# =========================
# CHARGEMENT & NETTOYAGE
# =========================

def load_and_clean(csv_path: str = CSV_PATH) -> duckdb.DuckDBPyRelation:
    """
    Lit le CSV avec DuckDB, applique le nettoyage des valeurs manquantes
    et la catégorisation MCC — sans charger tout en mémoire.
    """
    con = duckdb.connect()

    mcc_case = build_mcc_case()

    query = f"""
        SELECT
            transaction_id,
            transactions_type,
            transactions_currency,
            amount_usd,
            transactions_state,
            direction,
            user_id,
            created_date,

            -- Valeurs manquantes
            COALESCE(ea_cardholderpresence, 'unknown')   AS ea_cardholderpresence,
            COALESCE(ea_merchant_city,      'unknown_city')    AS ea_merchant_city,
            COALESCE(ea_merchant_country,   'unknown_country') AS ea_merchant_country,
            COALESCE(ea_merchant_mcc,       9999)              AS ea_merchant_mcc,

            -- Catégorie MCC
            {mcc_case} AS merchant_category

        FROM read_csv_auto('{csv_path}')

        -- Supprimer les lignes sans user_id ni date (équivalent dropna du notebook)
        WHERE user_id    IS NOT NULL
          AND created_date IS NOT NULL
    """

    print(f"📂 Lecture de {csv_path} avec DuckDB...")
    result = con.execute(query)
    print(f"✅ Données chargées et nettoyées")
    return con, result


# =========================
# FEATURE ENGINEERING PAR USER
# =========================

def build_user_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Agrège les transactions au niveau utilisateur.
    Reproduit fidèlement la cellule 38 du notebook.
    """
    print("⚙️  Construction des features par utilisateur...")

    df_user = con.execute("""
        WITH base AS (
            SELECT
                transaction_id,
                transactions_type,
                transactions_currency,
                amount_usd,
                transactions_state,
                direction,
                user_id,
                created_date,
                COALESCE(ea_cardholderpresence, 'unknown')         AS ea_cardholderpresence,
                COALESCE(ea_merchant_city,      'unknown_city')    AS ea_merchant_city,
                COALESCE(ea_merchant_country,   'unknown_country') AS ea_merchant_country,
                COALESCE(ea_merchant_mcc,       9999)              AS ea_merchant_mcc,
                merchant_category
            FROM cleaned
            WHERE user_id IS NOT NULL
              AND created_date IS NOT NULL
              AND transactions_state = 'COMPLETED'
        ),

        agg AS (
            SELECT
                user_id,

                SUM(amount_usd)                             AS total_spent,
                AVG(amount_usd)                             AS avg_transaction_amount,

                COUNT(*)                                    AS total_transactions,
                COUNT(*) FILTER (WHERE direction = 'INBOUND')  AS inbound_transactions,
                COUNT(*) FILTER (WHERE direction = 'OUTBOUND') AS outbound_transactions,

                COUNT(DISTINCT ea_merchant_mcc) FILTER (WHERE ea_merchant_mcc <> 9999)
                                                            AS unique_merchants,
                COUNT(*) FILTER (WHERE ea_merchant_mcc <> 9999)  AS merchant_transactions,

                COALESCE(
                    MODE(ea_merchant_country) FILTER (WHERE ea_merchant_country <> 'unknown_country'),
                    'unknown_country'
                )                                           AS top_merchant_country,

                MODE(transactions_currency)                 AS top_merchant_currency,

                COALESCE(
                    MODE(merchant_category) FILTER (WHERE merchant_category <> 'Unknown'),
                    'Unknown'
                )                                           AS top_merchant_category

            FROM base
            GROUP BY user_id
        )

            SELECT
            *,
            CASE
                WHEN total_transactions > 0
                THEN CAST(inbound_transactions AS FLOAT) / total_transactions
                ELSE 0
            END AS inbound_ratio,
            CASE
                WHEN total_transactions > 0
                THEN CAST(merchant_transactions AS FLOAT) / total_transactions
                ELSE 0
            END AS merchant_ratio
        FROM agg
    """).df()

    print(f"✅ {len(df_user):,} utilisateurs — {len(df_user.columns)} features construites")
    return df_user


# =========================
# PIPELINE PRINCIPAL
# =========================

def run(csv_path: str = CSV_PATH, output_path: str = OUTPUT_PATH) -> pd.DataFrame:
    """
    Lance le pipeline complet : lecture → nettoyage → features → export.
    """
    # Créer le dossier de sortie si besoin
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Connexion DuckDB
    con = duckdb.connect()

    # Créer une vue DuckDB pour réutiliser dans les requêtes
    mcc_case = build_mcc_case()
    con.execute(f"""
        CREATE VIEW cleaned AS
        SELECT
            *,
            {mcc_case} AS merchant_category
        FROM read_csv_auto('{csv_path}')
        WHERE user_id IS NOT NULL
          AND created_date IS NOT NULL
    """)

    # Construire les features utilisateur
    df_user = build_user_features(con)

    # Sauvegarder en parquet pour les étapes suivantes
    df_user.to_parquet(output_path, index=False)
    print(f"💾 Features sauvegardées → {output_path}")

    con.close()
    return df_user


# =========================
# POINT D'ENTRÉE
# =========================

if __name__ == "__main__":
    df_user = run()
    print("\n📊 Aperçu des features :")
    print(df_user.head())
    print("\n📐 Shape :", df_user.shape)
    print("\n📋 Colonnes :", df_user.columns.tolist())
