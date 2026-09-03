# -*- coding: utf-8 -*-
"""
============================================================================
 APP D'ESTIMATION IMMOBILIÈRE EN TEMPS RÉEL — Wafa Immobilier
============================================================================
Interface Streamlit qui charge le modèle final (Blend XGBoost + CatBoost)
entraîné dans modele_final_step2.py et permet à un utilisateur de saisir
les caractéristiques d'un bien pour obtenir une estimation de prix
instantanée.

Lancement :
    streamlit run app_estimation.py

Fichiers requis dans le même dossier :
    - xgboost_step2_final.pkl
    - catboost_step2_final.cbm
    - dataset_clean_step2_corrige.csv   (menus déroulants + comparaison marché,
      dégradation silencieuse si absent)
============================================================================
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor

# ============================================================================
# CONFIGURATION (doit rester identique à modele_final_step2.py)
# ============================================================================

POIDS_XGB = 0.30  # 30% XGBoost / 70% CatBoost
CATEGORICAL = ["city", "neighborhood", "property_type", "floor_appartement", "floor"]
FEATURES = [
    "city", "neighborhood", "neighborhood_is_generic", "property_type",
    "surface_m2", "bedrooms", "bathrooms", "floor",
    "floor_known", "surface_known", "rooms_info_known", "floor_appartement",
]

MODELE_XGB_PATH = "xgboost_step2_final.pkl"
MODELE_CB_PATH = "catboost_step2_final.cbm"
DATASET_PATH = "dataset_clean_step2_corrige.csv"

PROPERTY_TYPES = ["Appartement", "Maison", "Villa"]


# ============================================================================
# STYLE — palette et typographie
# ============================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Public+Sans:wght@400;500;600;700&display=swap');

:root {
    --emerald-900: #0F5C4A;
    --emerald-700: #16785F;
    --terracotta-600: #BD5B34;
    --terracotta-700: #A44B29;
    --gold-500: #C79A3D;
    --sand-50: #FBF6EC;
    --sand-100: #F3EAD6;
    --ink-800: #2B2420;
    --ink-600: #5C5347;
    --card: #FFFFFF;
}

html, body, [class*="st-"], [class*="css"] {
    font-family: 'Public Sans', sans-serif;
    color: var(--ink-800);
}
h1, h2, h3, .hero-title, .result-value {
    font-family: 'Fraunces', serif;
}

.stApp {
    background: var(--sand-50);
}
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; max-width: 780px; }

/* --- Barre de marque (logo + nom) --- */
.brand-bar {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.9rem;
}
.brand-badge {
    width: 34px;
    height: 34px;
    border-radius: 9px;
    background: linear-gradient(135deg, var(--terracotta-600), var(--gold-500));
    color: #FFFFFF;
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.15rem;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.brand-name {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: var(--emerald-900);
    letter-spacing: 0.01em;
}

/* --- Bannière hero --- */
.hero {
    background: linear-gradient(135deg, var(--emerald-900) 0%, var(--emerald-700) 100%);
    border-radius: 18px;
    padding: 2.2rem 2rem 1.6rem 2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    color: var(--gold-500);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-bottom: 0.3rem;
}
.hero-title {
    color: #FFFFFF;
    font-size: 2.1rem;
    font-weight: 600;
    line-height: 1.2;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    color: #E7DFC9;
    font-size: 0.98rem;
    max-width: 46ch;
}
.hero-pattern {
    height: 10px;
    margin-top: 1.4rem;
    border-radius: 4px;
    background-image:
        repeating-linear-gradient(45deg, transparent, transparent 7px, rgba(199,154,61,0.55) 7px, rgba(199,154,61,0.55) 8px),
        repeating-linear-gradient(-45deg, transparent, transparent 7px, rgba(199,154,61,0.55) 7px, rgba(199,154,61,0.55) 8px);
    background-color: rgba(255,255,255,0.06);
}

/* --- Sections --- */
.section-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--terracotta-600);
    letter-spacing: 0.03em;
    margin-bottom: 0.3rem;
    margin-top: 0.2rem;
}

/* Conteneurs bordés Streamlit (st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--sand-100) !important;
    border-radius: 14px !important;
}

/* --- Bouton principal --- */
.stButton > button[kind="primary"] {
    background: var(--terracotta-600);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 0.65rem 1rem;
    transition: background 0.15s ease;
}
.stButton > button[kind="primary"]:hover {
    background: var(--terracotta-700);
}

/* --- Carte résultat --- */
.result-card {
    background: linear-gradient(180deg, var(--sand-100) 0%, var(--sand-50) 100%);
    border-left: 4px solid var(--terracotta-600);
    border-radius: 12px;
    padding: 1.5rem 1.7rem;
    margin: 1rem 0;
}
.result-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--ink-600);
    letter-spacing: 0.03em;
    margin-bottom: 0.2rem;
}
.result-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--emerald-900);
    line-height: 1.1;
}
.result-sub {
    font-size: 0.92rem;
    color: var(--ink-600);
    margin-top: 0.3rem;
}

/* --- Note de comparaison marché --- */
.market-note {
    background: var(--card);
    border-left: 3px solid var(--gold-500);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    color: var(--ink-600);
    margin-top: 0.6rem;
}

/* --- Indicateur de confiance --- */
.confidence-note {
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.9rem;
    margin-top: 0.6rem;
    border-left: 3px solid;
}
.confidence-high {
    background: #EEF6F1;
    border-color: var(--emerald-700);
    color: #1D4536;
}
.confidence-medium {
    background: #FBF3E1;
    border-color: var(--gold-500);
    color: #6B4E12;
}
.confidence-low {
    background: #FBEEE7;
    border-color: var(--terracotta-600);
    color: #7A3418;
}

/* Metrics secondaires (XGBoost / CatBoost) */
div[data-testid="stMetric"] {
    background: var(--sand-50);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
}
div[data-testid="stMetricValue"] { color: var(--emerald-900); }

.footer-note {
    font-size: 0.82rem;
    color: var(--ink-600);
    text-align: center;
    margin-top: 1.6rem;
    line-height: 1.5;
}
</style>
"""


# ============================================================================
# CHARGEMENT (mis en cache pour ne pas recharger à chaque interaction)
# ============================================================================

@st.cache_resource
def load_models():
    pipeline_xgb = joblib.load(MODELE_XGB_PATH)
    modele_cb = CatBoostRegressor()
    modele_cb.load_model(MODELE_CB_PATH)
    return pipeline_xgb, modele_cb


@st.cache_data
def load_reference_data():
    try:
        return pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        return None


def get_neighborhoods(df: pd.DataFrame, city: str) -> list[str]:
    subset = df.loc[df["city"] == city, "neighborhood"].dropna().unique().tolist()
    return sorted(subset)


# ============================================================================
# CONSTRUCTION DE LA LIGNE À PRÉDIRE
# ============================================================================

def build_input_row(
    city: str,
    neighborhood: str,
    neighborhood_is_generic: bool,
    property_type: str,
    surface_m2: float,
    surface_known: bool,
    bedrooms: int,
    bathrooms: int,
    rooms_info_known: bool,
    floor_value: str,
) -> pd.DataFrame:
    floor_known = floor_value != ""
    floor_str = floor_value if floor_known else "unknown"
    floor_appartement_str = floor_str if property_type == "Appartement" else "unknown"

    row = {
        "city": city,
        "neighborhood": neighborhood.strip().lower() if neighborhood else "unknown",
        "neighborhood_is_generic": bool(neighborhood_is_generic),
        "property_type": property_type,
        "surface_m2": float(surface_m2),
        "bedrooms": float(bedrooms),
        "bathrooms": float(bathrooms),
        "floor": floor_str,
        "floor_known": int(floor_known),
        "surface_known": int(surface_known),
        "rooms_info_known": int(rooms_info_known),
        "floor_appartement": floor_appartement_str,
    }
    X = pd.DataFrame([row])
    for col in CATEGORICAL:
        X[col] = X[col].astype(str)
    return X[FEATURES]


def predict_price(pipeline_xgb, modele_cb, X: pd.DataFrame) -> dict:
    pred_xgb = float(np.maximum(pipeline_xgb.predict(X[FEATURES]), 0)[0])
    pred_cb = float(np.maximum(modele_cb.predict(X[FEATURES]), 0)[0])
    pred_blend = POIDS_XGB * pred_xgb + (1 - POIDS_XGB) * pred_cb
    return {"xgboost": pred_xgb, "catboost": pred_cb, "blend": pred_blend}


def format_dh(value: float) -> str:
    return f"{value:,.0f} DH".replace(",", " ")


def confidence_assessment(df_ref, city, neighborhood, property_type, preds) -> tuple[str, str]:
    """
    Évalue la fiabilité de l'estimation à partir de deux signaux :
    1. Le nombre d'annonces comparables (même ville + même quartier + même
       type de bien) dans le dataset d'entraînement.
    2. L'écart relatif entre XGBoost et CatBoost — un grand écart trahit une
       incertitude du modèle (cf. analyse sur le quartier "maarif").
    Retourne (niveau, message_html) où niveau ∈ {"high", "medium", "low"}.
    """
    n_comparables = 0
    if df_ref is not None and neighborhood:
        mask = (
            (df_ref["city"] == city)
            & (df_ref["property_type"] == property_type)
            & (df_ref["neighborhood"].str.lower() == neighborhood.strip().lower())
        )
        n_comparables = int(mask.sum())

    ecart_pct = abs(preds["xgboost"] - preds["catboost"]) / preds["blend"] if preds["blend"] else 0

    if n_comparables >= 30 and ecart_pct < 0.25:
        niveau = "high"
        msg = (f"✅ Estimation basée sur un échantillon solide "
               f"({n_comparables} annonces comparables pour ce quartier).")
    elif n_comparables >= 10 and ecart_pct < 0.5:
        niveau = "medium"
        msg = (f"⚠️ Échantillon limité ({n_comparables} annonces comparables pour ce quartier) "
               f"— estimation moins fiable, à recouper avec d'autres sources.")
    else:
        niveau = "low"
        raisons = []
        if n_comparables < 10:
            raisons.append(f"seulement {n_comparables} annonce(s) comparable(s) pour ce quartier")
        if ecart_pct >= 0.5:
            raisons.append(f"les deux modèles divergent fortement ({ecart_pct:.0%} d'écart)")
        msg = "🔴 Estimation incertaine : " + " et ".join(raisons) + ". À prendre avec prudence."

    return niveau, msg


# ============================================================================
# INTERFACE
# ============================================================================

def main():
    st.set_page_config(
        page_title="Wafa Immobilier — Estimation",
        page_icon="🏠",
        layout="centered",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="brand-bar">
            <div class="brand-badge">W</div>
            <div class="brand-name">Wafa Immobilier</div>
        </div>
        <div class="hero">
            <div class="hero-eyebrow">ESTIMATION IMMOBILIÈRE</div>
            <div class="hero-title">Estimez la valeur de votre bien</div>
            <div class="hero-sub">
                Renseignez quelques caractéristiques et obtenez une estimation
                de prix instantanée, basée sur les données du marché marocain.
            </div>
            <div class="hero-pattern"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        pipeline_xgb, modele_cb = load_models()
    except FileNotFoundError as e:
        st.error(
            f"Fichier modèle introuvable : {e}. "
            f"Place `{MODELE_XGB_PATH}` et `{MODELE_CB_PATH}` dans le même dossier que cette app."
        )
        st.stop()

    df_ref = load_reference_data()  # dégradation silencieuse si absent

    with st.container(border=True):
        st.markdown('<div class="section-label">TYPE ET LOCALISATION</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            property_type = st.selectbox("Type de bien", PROPERTY_TYPES)
        with col2:
            if df_ref is not None:
                villes = sorted(df_ref["city"].unique().tolist())
            else:
                villes = ["Casablanca", "Rabat", "Marrakech", "Tanger", "Agadir"]
            city = st.selectbox("Ville", villes)

        col3, col4 = st.columns([2, 1])
        with col3:
            if df_ref is not None:
                quartiers = get_neighborhoods(df_ref, city)
            else:
                quartiers = []
            mode_quartier = st.radio(
                "Quartier", ["Choisir dans la liste", "Saisir librement"],
                horizontal=True, label_visibility="collapsed",
            )
            if mode_quartier == "Choisir dans la liste" and quartiers:
                neighborhood = st.selectbox("Quartier", quartiers)
                neighborhood_is_generic = False
            else:
                neighborhood = st.text_input("Nom du quartier", value="")
                neighborhood_is_generic = st.checkbox("Localisation générique", value=False)
        with col4:
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-label">SURFACE ET PIÈCES</div>', unsafe_allow_html=True)
        col5, col6 = st.columns(2)
        with col5:
            surface_connue = st.checkbox("Surface connue", value=True)
            surface_m2 = st.number_input(
                "Surface (m²)", min_value=10.0, max_value=5000.0, value=90.0, step=5.0,
                disabled=not surface_connue,
            )
            if not surface_connue:
                surface_m2 = (
                    float(df_ref[df_ref["property_type"] == property_type]["surface_m2"].median())
                    if df_ref is not None else 90.0
                )
        with col6:
            rooms_info_known = st.checkbox("Infos pièces fiables", value=True)

        col7, col8 = st.columns(2)
        with col7:
            bedrooms = st.number_input("Chambres", min_value=1, max_value=15, value=3, step=1)
        with col8:
            bathrooms = st.number_input("Salles de bain", min_value=0, max_value=12, value=1, step=1)

        etage_connu = st.checkbox("Étage connu", value=(property_type == "Appartement"))
        if etage_connu:
            etage_num = st.number_input("Numéro d'étage (0 = rez-de-chaussée)", min_value=0, max_value=30, value=1, step=1)
            floor_value = str(etage_num)
        else:
            floor_value = ""

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    estimer = st.button("💰 Estimer le prix", type="primary", use_container_width=True)

    if estimer:
        X = build_input_row(
            city=city, neighborhood=neighborhood, neighborhood_is_generic=neighborhood_is_generic,
            property_type=property_type, surface_m2=surface_m2, surface_known=surface_connue,
            bedrooms=bedrooms, bathrooms=bathrooms, rooms_info_known=rooms_info_known,
            floor_value=floor_value,
        )
        preds = predict_price(pipeline_xgb, modele_cb, X)
        prix_m2 = preds["blend"] / surface_m2 if surface_m2 > 0 else 0

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">ESTIMATION</div>
                <div class="result-value">{format_dh(preds['blend'])}</div>
                <div class="result-sub">Soit environ {format_dh(prix_m2)} / m²</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        niveau, msg_confiance = confidence_assessment(df_ref, city, neighborhood, property_type, preds)
        st.markdown(f'<div class="confidence-note confidence-{niveau}">{msg_confiance}</div>', unsafe_allow_html=True)

        if df_ref is not None:
            comparables = df_ref[(df_ref["city"] == city) & (df_ref["property_type"] == property_type)]
            if len(comparables) >= 5:
                prix_m2_marche = (comparables["price_dh"] / comparables["surface_m2"]).median()
                st.markdown(
                    f"""<div class="market-note">
                        Prix médian du marché pour un(e) {property_type.lower()} à {city} :
                        <strong>{format_dh(prix_m2_marche)}/m²</strong> (sur {len(comparables)} annonces comparables)
                    </div>""",
                    unsafe_allow_html=True,
                )

        with st.expander("Détail technique de la prédiction"):
            c1, c2 = st.columns(2)
            c1.metric("XGBoost", format_dh(preds["xgboost"]))
            c2.metric("CatBoost", format_dh(preds["catboost"]))
            st.caption(f"Blend : {POIDS_XGB:.0%} XGBoost / {1 - POIDS_XGB:.0%} CatBoost")
            st.dataframe(X.T.rename(columns={0: "Valeur"}))

    st.markdown(
        """
        <div class="footer-note">
            Cette estimation est générée par un modèle statistique entraîné sur des annonces
            du marché marocain. Elle ne remplace pas une expertise immobilière.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()