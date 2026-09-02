"""
app.py
------
Interface Streamlit du pipeline de segmentation client NexBank.

Lancer avec :
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="Segmentation client — NexBank", page_icon="🏦", layout="wide")
import theme
theme.apply()


# =========================
# CHARGEMENT (mis en cache)
# =========================
@st.cache_resource(show_spinner="Chargement du modèle et des données...")
def get_resources():
    return core.load_resources()


try:
    kmeans, scaler, feature_columns, df_clustered = get_resources()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# =========================
# EN-TÊTE
# =========================
st.title("Segmentation client — NexBank")
st.caption(
    f"Chaque client est décrit par {len(feature_columns)} indicateurs "
    "(montants, fréquence, pays et catégories de commerçants)."
)

col1, col2, col3 = st.columns(3)
col1.metric("Utilisateurs profilés", f"{len(df_clustered):,}".replace(",", " "))
col2.metric("Segments", df_clustered["segment"].nunique())
col3.metric("Transactions analysées", "2,7 M")

vue, recherche, simulation = st.tabs(
    ["Vue d'ensemble", "Chercher un client", "Simuler un nouveau client"]
)


# =========================
# ONGLET 1 — VUE D'ENSEMBLE
# =========================
with vue:
    summary = core.segments_summary(df_clustered)

    st.subheader("Les segments en un coup d'œil")
    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "segment": "Segment",
            "nb_utilisateurs": st.column_config.NumberColumn("Utilisateurs", format="%d"),
            "depense_moyenne": st.column_config.NumberColumn("Dépense moyenne", format="%.2f €"),
            "panier_moyen": st.column_config.NumberColumn("Panier moyen", format="%.2f €"),
            "nb_transactions_moyen": st.column_config.NumberColumn("Transactions moy.", format="%.1f"),
        },
    )

    gauche, droite = st.columns(2)
    with gauche:
        st.subheader("Répartition des utilisateurs")
        st.bar_chart(summary.set_index("segment")["nb_utilisateurs"], height=320)
    with droite:
        st.subheader("Panier moyen par segment")
        st.bar_chart(summary.set_index("segment")["panier_moyen"], height=320)

    st.subheader("Dépense totale et panier moyen")
    echantillon = df_clustered.sample(min(3000, len(df_clustered)), random_state=42)
    st.scatter_chart(
        echantillon,
        x="avg_transaction_amount",
        y="total_spent",
        color="segment",
        height=420,
    )
    st.caption("Échantillon de 3 000 utilisateurs pour garder le graphique lisible.")

    for segment, action in core.SEGMENT_ACTIONS.items():
        if segment in set(df_clustered["segment"]):
            st.markdown(f"**{segment}** — {action}")


# =========================
# ONGLET 2 — RECHERCHE
# =========================
with recherche:
    st.subheader("Segment d'un client existant")
    st.caption("Choisissez un client dans la liste, ou tirez-en un au hasard.")

    filtre = st.selectbox(
        "Filtrer par segment",
        ["Tous"] + sorted(df_clustered["segment"].unique()),
    )
    sous_ensemble = (
        df_clustered if filtre == "Tous"
        else df_clustered[df_clustered["segment"] == filtre]
    )
    ids = sous_ensemble["user_id"].tolist()

    if st.button("Tirer un client au hasard"):
        st.session_state["user_id"] = sous_ensemble["user_id"].sample(1).iloc[0]

    choisi = st.session_state.get("user_id")
    user_id = st.selectbox(
        "Identifiant client",
        options=ids,
        index=ids.index(choisi) if choisi in ids else 0,
        help="Tapez quelques caractères pour filtrer la liste.",
    )

    infos = core.get_user_segment(df_clustered, user_id)
    st.markdown(theme.segment_badge(infos["segment"], infos["description"]),
            unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.metric("Dépense totale", f"{infos['total_spent']:,.2f}".replace(",", " "))
    b.metric("Panier moyen", f"{infos['avg_transaction']:,.2f}".replace(",", " "))
    c.metric("Transactions", infos["total_transactions"])

    st.markdown(
        f"Pays marchand principal : **{infos['top_country']}** · "
        f"Catégorie principale : **{infos['top_category']}**"
    )
    st.info(core.SEGMENT_ACTIONS.get(infos["segment"], ""))


# =========================
# ONGLET 3 — SIMULATION
# =========================
with simulation:
    st.subheader("Prédire le segment d'un nouveau client")
    st.caption("Renseignez le profil de transactions : le modèle attribue un segment.")

    gauche, droite = st.columns(2)
    with gauche:
        total_spent = st.number_input("Dépense totale (€)", min_value=0.0, value=15000.0, step=500.0)
        avg_amount = st.number_input("Montant moyen par transaction (€)", min_value=0.0, value=250.0, step=10.0)
        total_tx = st.number_input("Nombre de transactions", min_value=0, value=60, step=1)
        unique_merchants = st.number_input("Marchands distincts", min_value=0, value=8, step=1)
    with droite:
        inbound_tx = st.number_input("Transactions entrantes", min_value=0, value=10, step=1)
        pays = st.selectbox(
            "Pays marchand principal", core.category_options(df_clustered, "top_merchant_country")
        )
        devise = st.selectbox(
            "Devise principale", core.category_options(df_clustered, "top_merchant_currency")
        )
        categorie = st.selectbox(
            "Catégorie marchande principale", core.category_options(df_clustered, "top_merchant_category")
        )

    outbound_tx = max(total_tx - inbound_tx, 0)
    inbound_ratio = round(inbound_tx / total_tx, 4) if total_tx else 0.0
    st.caption(f"Transactions sortantes déduites : {outbound_tx} · ratio entrant : {inbound_ratio}")

    if st.button("Prédire le segment", type="primary"):
        features = {
            "total_spent": float(total_spent),
            "avg_transaction_amount": float(avg_amount),
            "total_transactions": int(total_tx),
            "inbound_transactions": int(inbound_tx),
            "outbound_transactions": int(outbound_tx),
            "unique_merchants": int(unique_merchants),
            "inbound_ratio": float(inbound_ratio),
            "top_merchant_country": pays,
            "top_merchant_currency": devise,
            "top_merchant_category": categorie,
        }
        resultat = core.predict_segment(features, kmeans, scaler, feature_columns, df_clustered)

        st.success(f"{resultat['segment']} (cluster {resultat['cluster_id']})")
        st.write(resultat["description"])
        st.info(resultat["action"])

        with st.expander("Voir les données envoyées au modèle"):
            st.dataframe(pd.DataFrame([features]).T.rename(columns={0: "valeur"}))
