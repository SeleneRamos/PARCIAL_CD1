import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import NotFittedError

# ══════════════════════════════════════════════════════════════════════════════
# CLASES Y FUNCIONES DEL PREPROCESAMIENTO
# Deben estar en el módulo raíz ANTES de cualquier joblib.load
# ══════════════════════════════════════════════════════════════════════════════

REGION_MAPPING = {
    'United-States': 'EEUU', 'Canada': 'Norteamerica_Otros',
    'Outlying-US(Guam-USVI-etc)': 'Norteamerica_Otros',
    'Mexico': 'Latinoamerica', 'Puerto-Rico': 'Latinoamerica',
    'El-Salvador': 'Latinoamerica', 'Cuba': 'Latinoamerica',
    'Jamaica': 'Latinoamerica', 'Dominican-Republic': 'Latinoamerica',
    'Guatemala': 'Latinoamerica', 'Columbia': 'Latinoamerica',
    'Haiti': 'Latinoamerica', 'Nicaragua': 'Latinoamerica',
    'Peru': 'Latinoamerica', 'Ecuador': 'Latinoamerica',
    'Trinadad&Tobago': 'Latinoamerica', 'Honduras': 'Latinoamerica',
    'Argentina': 'Latinoamerica', 'Chile': 'Latinoamerica',
    'Brasil': 'Latinoamerica', 'Venezuela': 'Latinoamerica',
    'Germany': 'Europa', 'England': 'Europa', 'Italy': 'Europa',
    'Poland': 'Europa', 'Portugal': 'Europa', 'Greece': 'Europa',
    'France': 'Europa', 'Ireland': 'Europa', 'Yugoslavia': 'Europa',
    'Scotland': 'Europa', 'Hungary': 'Europa', 'Holand-Netherlands': 'Europa',
    'Philippines': 'Asia', 'India': 'Asia', 'China': 'Asia', 'Japan': 'Asia',
    'Vietnam': 'Asia', 'Taiwan': 'Asia', 'Iran': 'Asia', 'Hong Kong': 'Asia',
    'Hong-Kong': 'Asia', 'Thailand': 'Asia', 'Cambodia': 'Asia',
    'Laos': 'Asia', 'South-Korea': 'Asia',
}

# education string → education-num (mismo mapeo del dataset UCI)
EDU_TO_NUM = {
    'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
    '9th': 5, '10th': 6, '11th': 7, '12th': 8,
    'HS-grad': 9, 'Some-college': 10, 'Assoc-acdm': 11,
    'Assoc-voc': 12, 'Bachelors': 13, 'Masters': 14,
    'Prof-school': 15, 'Doctorate': 16,
}

# Etiquetas amigables para el selector
EDU_LABELS = [
    'Preschool', '1st-4th', '5th-6th', '7th-8th', '9th',
    '10th', '11th', '12th', 'HS-grad', 'Some-college',
    'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Masters',
    'Prof-school', 'Doctorate',
]

def regionalizar_paises(X):
    X_copy = X.copy()
    X_copy['region'] = X_copy['native-country'].map(REGION_MAPPING)
    X_copy = X_copy.drop('native-country', axis=1)
    return X_copy

def agrupar_educacion(X):
    X_copy = X.copy()
    edu_map = {
        'Preschool': 'Basica_Incompleta', '1st-4th': 'Basica_Incompleta',
        '5th-6th': 'Basica_Incompleta', '7th-8th': 'Basica_Incompleta',
        '9th': 'Basica_Incompleta', '10th': 'Basica_Incompleta',
        '11th': 'Basica_Incompleta', '12th': 'Basica_Incompleta',
        'HS-grad': 'Secundaria_Completa',
        'Some-college': 'Tecnica_o_Universitaria_Incompleta',
        'Assoc-voc': 'Tecnica_o_Universitaria_Incompleta',
        'Assoc-acdm': 'Tecnica_o_Universitaria_Incompleta',
        'Bachelors': 'Universitaria', 'Masters': 'Maestría',
        'Prof-school': 'Escuela profesional', 'Doctorate': 'Doctorado',
    }
    X_copy['nivel_educativo'] = X_copy['education'].map(edu_map)
    X_copy = X_copy.drop([c for c in ['education', 'education-num'] if c in X_copy.columns], axis=1)
    return X_copy

def agrupar_ocupacion(X):
    X_copy = X.copy()
    ocu_map = {
        'Exec-managerial': 'Administrativo_Especializado',
        'Prof-specialty':  'Administrativo_Especializado',
        'Tech-support':    'Administrativo_Especializado',
        'Adm-clerical':    'Administrativo_Especializado',
        'Sales':           'Administrativo_Especializado',
        'Craft-repair':      'Operativo_Manual',
        'Machine-op-inspct': 'Operativo_Manual',
        'Transport-moving':  'Operativo_Manual',
        'Handlers-cleaners': 'Operativo_Manual',
        'Farming-fishing':   'Operativo_Manual',
        'Protective-serv': 'Servicios',
        'Other-service':   'Servicios',
        'Priv-house-serv': 'Servicios',
        'Armed-Forces': 'Otros_Militar',
    }
    X_copy['perfil_ocupacional'] = X_copy['occupation'].map(ocu_map)
    X_copy = X_copy.drop('occupation', axis=1)
    return X_copy

def imputar_desconocido(X):
    X_copy = X.copy()
    for col in ['workclass', 'region', 'perfil_ocupacional']:
        if col in X_copy.columns:
            X_copy[col] = X_copy[col].fillna('Desconocido')
    return X_copy

def log1p_capital(X):
    X_copy = X.copy()
    X_copy['log_capital_gain'] = np.log1p(X_copy['capital-gain'])
    X_copy['log_capital_loss'] = np.log1p(X_copy['capital-loss'])
    return X_copy.drop(columns=['capital-gain', 'capital-loss'])


class DiscretizadorSupervisado(BaseEstimator, TransformerMixin):
    def __init__(self, max_depth=3, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("Requiere y.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.arboles_ = []
        for col in X.columns:
            t = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            t.fit(X[[col]], y)
            self.arboles_.append(t)
        return self

    def transform(self, X, y=None):
        if not hasattr(self, 'arboles_'):
            raise NotFittedError("No entrenado.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_copy = X.copy()
        for i, col in enumerate(X.columns):
            X_copy[col] = self.arboles_[i].predict_proba(X[[col]])[:, 1]
        return X_copy.to_numpy()


class DiscretizadorCapital(BaseEstimator, TransformerMixin):
    def __init__(self, max_depth=3, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y=None):
        self._disc = DiscretizadorSupervisado(self.max_depth, self.random_state)
        self._disc.fit(X[['capital-gain', 'capital-loss']], y)
        proba = self._disc.transform(X[['capital-gain', 'capital-loss']])
        proba_df = pd.DataFrame(proba, columns=['capital-gain', 'capital-loss'])
        self.mapeos_ = {
            col: {v: f"Nivel_{i+1}" for i, v in enumerate(sorted(proba_df[col].unique()))}
            for col in ['capital-gain', 'capital-loss']
        }
        return self

    def transform(self, X, y=None):
        X_copy = X.copy().reset_index(drop=True)
        proba = self._disc.transform(X_copy[['capital-gain', 'capital-loss']])
        proba_df = pd.DataFrame(proba, columns=['capital-gain', 'capital-loss'])
        X_copy['cat_capital-gain'] = proba_df['capital-gain'].map(self.mapeos_['capital-gain'])
        X_copy['cat_capital-loss'] = proba_df['capital-loss'].map(self.mapeos_['capital-loss'])
        return X_copy.drop(columns=['capital-gain', 'capital-loss'])


class CastCategorias(BaseEstimator, TransformerMixin):
    """Fija las categorías aprendidas en train para XGBoost nativo."""
    def __init__(self, cat_cols):
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        self.categorias_ = {c: pd.unique(X[c].dropna()) for c in self.cat_cols}
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = pd.Categorical(X[c], categories=self.categorias_[c])
        return X


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Census Income Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --bg-main:      #0d0f14;
    --bg-card:      #161922;
    --bg-card2:     #1c2030;
    --border:       #2a2f3e;
    --accent:       #00e5ff;
    --accent2:      #7c3aed;
    --accent3:      #f59e0b;
    --text-primary: #eef0f7;
    --text-secondary:#c8cfe0;
    --text-muted:   #8892a4;
    --green:        #10b981;
    --red:          #ef4444;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-main) !important;
    font-family: var(--sans);
    color: var(--text-primary);
}
[data-testid="stSidebar"] {
    background-color: #111318 !important;
    border-right: 1px solid var(--border);
}

/* ── Hero ──────────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0d0f14 0%, #161030 50%, #0d1520 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(0,229,255,.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag   { font-family:var(--mono);font-size:.7rem;letter-spacing:.2em;color:var(--accent);text-transform:uppercase;margin-bottom:.5rem; }
.hero-title { font-family:var(--mono);font-size:2rem;font-weight:700;color:var(--text-primary);margin:0 0 .5rem;line-height:1.2; }
.hero-title span { color:var(--accent); }
.hero-sub   { color:var(--text-secondary);font-size:.95rem;margin:0; }

/* ── Cards ─────────────────────────────────────────────────────────────────── */
.card       { background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.5rem;margin-bottom:1rem; }
.card-title { font-family:var(--mono);font-size:.72rem;letter-spacing:.15em;color:var(--text-muted);text-transform:uppercase;margin-bottom:1rem;padding-bottom:.75rem;border-bottom:1px solid var(--border); }

/* ── Result cards ───────────────────────────────────────────────────────────── */
.result-card     { border-radius:14px;padding:1.5rem 1.75rem;margin-bottom:1rem;position:relative;overflow:hidden; }
.result-positive { background:linear-gradient(135deg,rgba(16,185,129,.14) 0%,rgba(16,185,129,.05) 100%);border:1px solid rgba(16,185,129,.4); }
.result-negative { background:linear-gradient(135deg,rgba(239,68,68,.12) 0%,rgba(239,68,68,.04) 100%);border:1px solid rgba(239,68,68,.35); }

.result-model-name { font-family:var(--mono);font-size:.75rem;letter-spacing:.12em;text-transform:uppercase;color:var(--text-muted);margin-bottom:.4rem; }
.result-verdict    { font-size:1.5rem;font-weight:600;margin-bottom:.3rem; }
.result-verdict.positive { color:#34d399; }
.result-verdict.negative { color:#f87171; }
.result-prob       { font-family:var(--mono);font-size:.85rem;color:var(--text-secondary); }
.prob-value        { color:var(--text-primary);font-weight:700; }

.prob-bar-container { margin-top:.75rem;height:6px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden; }
.prob-bar-fill      { height:100%;border-radius:99px; }
.prob-bar-positive  { background:linear-gradient(90deg,#10b981,#6ee7b7); }
.prob-bar-negative  { background:linear-gradient(90deg,#ef4444,#fca5a5); }

/* ── Consensus ──────────────────────────────────────────────────────────────── */
.consensus       { background:var(--bg-card2);border:1px solid var(--border);border-radius:12px;padding:1.25rem 1.5rem;text-align:center;margin-top:1rem; }
.consensus-label { font-family:var(--mono);font-size:.65rem;letter-spacing:.2em;color:var(--text-muted);text-transform:uppercase;margin-bottom:.4rem; }
.consensus-value { font-size:1.1rem;font-weight:600; }
.consensus-sub   { font-family:var(--mono);font-size:.78rem;color:var(--text-muted);margin-top:.3rem; }
.consensus-sub span { color:var(--text-primary);font-weight:700; }

/* ── Chips ──────────────────────────────────────────────────────────────────── */
.chip        { display:inline-block;padding:.2rem .6rem;border-radius:99px;font-family:var(--mono);font-size:.65rem;letter-spacing:.08em;margin-right:.35rem;margin-bottom:.25rem; }
.chip-cyan   { background:rgba(0,229,255,.12);  color:#67eeff; border:1px solid rgba(0,229,255,.3); }
.chip-purple { background:rgba(124,58,237,.18); color:#c4b5fd; border:1px solid rgba(124,58,237,.4); }
.chip-amber  { background:rgba(245,158,11,.14); color:#fcd34d; border:1px solid rgba(245,158,11,.3); }

/* ── Section label ──────────────────────────────────────────────────────────── */
.section-label { font-family:var(--mono);font-size:.68rem;letter-spacing:.18em;color:var(--text-muted);text-transform:uppercase;margin-bottom:.6rem;margin-top:1.4rem; }

/* ── Pipeline info card ──────────────────────────────────────────────────────── */
.pipe-row    { margin-bottom:.75rem; }
.pipe-desc   { color:var(--text-muted);font-size:.77rem;margin-top:.2rem;line-height:1.5; }
.pipe-name   { color:var(--text-secondary);font-size:.82rem; }

/* ── Streamlit widget overrides ─────────────────────────────────────────────── */
.stSelectbox > label,
.stNumberInput > label,
.stSlider > label {
    font-family: var(--sans) !important;
    font-size: .82rem !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div {
    background-color: #1a1d27 !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"]  > div:focus-within {
    border-color: var(--accent) !important;
}
/* Dropdown list items */
li[role="option"] {
    background-color: #1a1d27 !important;
    color: var(--text-primary) !important;
}
li[role="option"]:hover,
li[aria-selected="true"] {
    background-color: #252a3a !important;
    color: #ffffff !important;
}
/* Number input text */
input[type="number"] {
    color: var(--text-primary) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #5b21b6) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .65rem 2rem !important;
    font-family: var(--mono) !important;
    font-size: .8rem !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(124,58,237,.4) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ─── PYTORCH BUILDER ──────────────────────────────────────────────────────────
def build_tuned_model_pytorch(capas, dropout, input_dim):
    layers_list, prev = [], input_dim
    for n in capas:
        layers_list += [nn.Linear(prev, n), nn.ReLU(),
                        nn.BatchNorm1d(n), nn.Dropout(dropout)]
        prev = n
    layers_list += [nn.Linear(prev, 1)]
    return nn.Sequential(*layers_list)


# ─── LOAD ASSETS ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    assets = {}
    base   = "modelo_final"
    errors = []

    for name in ("pipeline_log", "pipeline_disc"):
        try:
            assets[name] = joblib.load(f"{base}/{name}.pkl")
        except Exception as e:
            errors.append(f"{name}.pkl → {e}")

    try:
        import tensorflow as tf
        assets["keras"]    = tf.keras.models.load_model(f"{base}/modelo_keras_final.keras")
        assets["keras_ok"] = True
    except Exception as e:
        errors.append(f"Keras → {e}")
        assets["keras_ok"] = False

    try:
        arch = joblib.load(f"{base}/arquitectura_pytorch.pkl")
        m    = build_tuned_model_pytorch(arch["capas"], arch["dropout"], arch["input_dim"])
        m.load_state_dict(torch.load(f"{base}/modelo_pytorch_final.pt", map_location="cpu"))
        m.eval()
        assets["pytorch"]    = m
        assets["pytorch_ok"] = True
    except Exception as e:
        errors.append(f"PyTorch → {e}")
        assets["pytorch_ok"] = False

    try:
        xgb_m = XGBClassifier()
        xgb_m.load_model(f"{base}/modelo_xgboost_final.json")
        assets["xgboost"]    = xgb_m
        assets["xgb_prepro"] = joblib.load(f"{base}/preprocesador_xgboost.pkl")
        assets["xgboost_ok"] = True
    except Exception as e:
        errors.append(f"XGBoost → {e}")
        assets["xgboost_ok"] = False

    assets["errors"] = errors
    return assets


# ─── PREDICT ──────────────────────────────────────────────────────────────────
def predict_all(row_df, assets):
    results = {}

    # ── Keras ─────────────────────────────────────────────────────────────────
    if assets.get("keras_ok") and "pipeline_log" in assets:
        try:
            X    = assets["pipeline_log"].transform(row_df).astype(np.float32)
            prob = float(assets["keras"].predict(X, verbose=0).ravel()[0])
            results["Keras (Red Neuronal)"] = {
                "prob": prob, "pred": int(prob >= 0.5),
                "pipeline": "pipeline_log", "color": "cyan",
            }
        except Exception as e:
            results["Keras (Red Neuronal)"] = {"error": str(e)}

    # ── PyTorch ───────────────────────────────────────────────────────────────
    if assets.get("pytorch_ok") and "pipeline_log" in assets:
        try:
            X = assets["pipeline_log"].transform(row_df).astype(np.float32)
            with torch.no_grad():
                prob = float(torch.sigmoid(
                    assets["pytorch"](torch.tensor(X))).item())
            results["PyTorch (Red Neuronal)"] = {
                "prob": prob, "pred": int(prob >= 0.5),
                "pipeline": "pipeline_log", "color": "purple",
            }
        except Exception as e:
            results["PyTorch (Red Neuronal)"] = {"error": str(e)}

    # ── XGBoost — encoding NATIVO (solo cast, sin OHE) ────────────────────────
    # El modelo fue entrenado con enable_categorical=True y CastCategorias,
    # que genera exactamente 12 columnas (5 num + 7 cat como dtype category).
    # Pasarle el OHE genera 55 columnas → Feature shape mismatch.
    if assets.get("xgboost_ok"):
        try:
            prepro   = assets["xgb_prepro"]
            cast     = prepro["cast"]       # CastCategorias
            num_cols = prepro["num_cols"]   # ['age','hours-per-week','capital-gain','capital-loss','education-num']
            cat_cols = prepro["cat_cols"]   # ['workclass','marital-status','occupation','relationship','race','sex','region']

            X = row_df.copy()
            # Regionalizar native-country → region
            X['region'] = X['native-country'].map(REGION_MAPPING).fillna('Otro')
            X = X.drop('native-country', axis=1)

            # Seleccionar solo las columnas que el modelo conoce (mismo orden que en training)
            X = X[num_cols + cat_cols]

            # Cast a dtype category (lo que XGBoost nativo espera)
            X_cast = cast.transform(X)

            prob = float(assets["xgboost"].predict_proba(X_cast)[:, 1][0])
            results["XGBoost"] = {
                "prob": prob, "pred": int(prob >= 0.5),
                "pipeline": "nativo (category)", "color": "amber",
            }
        except Exception as e:
            results["XGBoost"] = {"error": str(e)}

    return results


# ─── UI ───────────────────────────────────────────────────────────────────────
assets = load_assets()

# Hero
st.markdown("""
<div class="hero">
  <div class="hero-tag">// Census Income · UCI Adult Dataset</div>
  <h1 class="hero-title">Income <span>Predictor</span></h1>
  <p class="hero-sub">Predicción multi-modelo del nivel de ingresos anual &nbsp;·&nbsp; Keras &nbsp;·&nbsp; PyTorch &nbsp;·&nbsp; XGBoost</p>
</div>
""", unsafe_allow_html=True)

def status_chip(ok, name, cls):
    icon  = "●" if ok else "○"
    label = "Cargado" if ok else "No encontrado"
    return f'<span class="chip {cls}">{icon} {name} — {label}</span>'

st.markdown(
    status_chip(assets.get("keras_ok"),   "Keras",   "chip-cyan")   +
    status_chip(assets.get("pytorch_ok"), "PyTorch", "chip-purple") +
    status_chip(assets.get("xgboost_ok"), "XGBoost", "chip-amber"),
    unsafe_allow_html=True,
)

if assets.get("errors"):
    with st.expander("⚠️ Advertencias de carga", expanded=False):
        for e in assets["errors"]:
            st.code(e, language="text")

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1.1, 1], gap="large")

# ─── LEFT — formulario ────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">// Datos del individuo</div>', unsafe_allow_html=True)

    # ── Personal ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Información Personal</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age  = st.number_input("Edad", min_value=17, max_value=90, value=35)
        sex  = st.selectbox("Sexo", ["Male", "Female"])
        race = st.selectbox("Raza", ["White", "Black", "Asian-Pac-Islander",
                                      "Amer-Indian-Eskimo", "Other"])
    with c2:
        # Selector de nivel educativo como categoría; se convierte a num internamente
        education_str  = st.selectbox("Nivel educativo", EDU_LABELS, index=12)  # Bachelors por defecto
        education_num  = EDU_TO_NUM[education_str]   # derivado automáticamente

        marital_status = st.selectbox("Estado civil", [
            "Married-civ-spouse", "Never-married", "Divorced",
            "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
        relationship   = st.selectbox("Relación familiar", [
            "Husband", "Not-in-family", "Own-child",
            "Unmarried", "Wife", "Other-relative"])

    # ── Laboral ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Información Laboral</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        workclass  = st.selectbox("Sector laboral", [
            "Private", "Self-emp-not-inc", "Self-emp-inc",
            "Federal-gov", "Local-gov", "State-gov",
            "Without-pay", "Never-worked"])
        occupation = st.selectbox("Ocupación", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
            "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
            "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"])
    with c4:
        hours_per_week = st.number_input("Horas / semana", min_value=1, max_value=99, value=40)
        native_country = st.selectbox("País de origen", [
            "United-States","Mexico","Philippines","Germany","Canada",
            "Puerto-Rico","El-Salvador","India","Cuba","England",
            "Jamaica","China","Italy","Dominican-Republic","Vietnam",
            "Guatemala","Japan","Poland","Columbia","Taiwan","Haiti",
            "Iran","Portugal","Nicaragua","Peru","France","Greece",
            "Ecuador","Ireland","Hong-Kong","Trinadad&Tobago","Cambodia",
            "Laos","Thailand","Yugoslavia","Outlying-US(Guam-USVI-etc)",
            "Honduras","Hungary","Scotland","Holand-Netherlands"])

    # ── Capital ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Capital Financiero</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        capital_gain = st.number_input("Capital gain ($)", min_value=0, max_value=9999999, value=0)
    with c6:
        capital_loss = st.number_input("Capital loss ($)", min_value=0, max_value=9999999, value=0)

    st.markdown("</div>", unsafe_allow_html=True)

    # Mostrar el num derivado como info
    st.markdown(
        f'<div style="font-family:\'Space Mono\',monospace;font-size:.72rem;color:#8892a4;'
        f'margin-bottom:.75rem;margin-top:-.25rem;">education-num derivado → '
        f'<span style="color:#c4b5fd">{education_num}</span></div>',
        unsafe_allow_html=True
    )

    predict_btn = st.button("🔍  PREDECIR INGRESO", use_container_width=True)


# ─── RIGHT — resultados ───────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-label">Resultados por Modelo</div>', unsafe_allow_html=True)

    if predict_btn:
        row = pd.DataFrame([{
            "age":            age,
            "workclass":      workclass,
            "education":      education_str,
            "education-num":  education_num,
            "marital-status": marital_status,
            "occupation":     occupation,
            "relationship":   relationship,
            "race":           race,
            "sex":            sex,
            "capital-gain":   capital_gain,
            "capital-loss":   capital_loss,
            "hours-per-week": hours_per_week,
            "native-country": native_country,
        }])

        with st.spinner("Ejecutando modelos..."):
            predictions = predict_all(row, assets)

        if not predictions:
            st.error("No hay modelos cargados. Verifica la carpeta `modelo_final/`.")
        else:
            chip_map = {"cyan":"chip-cyan","purple":"chip-purple","amber":"chip-amber"}
            votes_pos, votes_neg, probs_all = 0, 0, []

            for model_name, res in predictions.items():
                if "error" in res:
                    st.markdown(f"""
                    <div class="result-card result-negative">
                      <div class="result-model-name">{model_name}</div>
                      <div style="color:#fca5a5;font-size:.82rem;margin-top:.25rem;">⚠ {res['error']}</div>
                    </div>""", unsafe_allow_html=True)
                    continue

                prob   = res["prob"]
                pred   = res["pred"]
                pipe   = res["pipeline"]
                color  = res["color"]
                is_pos = pred == 1
                probs_all.append(prob)
                if is_pos: votes_pos += 1
                else:       votes_neg += 1

                card_cls = "result-positive" if is_pos else "result-negative"
                verd_cls = "positive"        if is_pos else "negative"
                verdict  = ">50K — Ingresos Altos" if is_pos else "≤50K — Ingresos Bajos"
                bar_cls  = "prob-bar-positive"      if is_pos else "prob-bar-negative"
                icon     = "💰" if is_pos else "📉"

                st.markdown(f"""
                <div class="result-card {card_cls}">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                      <div class="result-model-name">{model_name}
                        <span class="chip {chip_map[color]}" style="margin-left:.4rem">{pipe}</span>
                      </div>
                      <div class="result-verdict {verd_cls}">{verdict}</div>
                      <div class="result-prob">P(&gt;50K) = <span class="prob-value">{prob:.1%}</span></div>
                    </div>
                    <div style="font-size:2rem;opacity:.75">{icon}</div>
                  </div>
                  <div class="prob-bar-container">
                    <div class="prob-bar-fill {bar_cls}" style="width:{int(prob*100)}%"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # ── Consenso ──────────────────────────────────────────────────────
            total = votes_pos + votes_neg
            if total > 0:
                avg_prob = float(np.mean(probs_all))
                if votes_pos > votes_neg:
                    cc, ct = "#34d399", f"✅ Consenso: &gt;50K ({votes_pos}/{total} modelos)"
                elif votes_neg > votes_pos:
                    cc, ct = "#f87171", f"❌ Consenso: ≤50K ({votes_neg}/{total} modelos)"
                else:
                    cc, ct = "#fcd34d", "⚖️ Empate entre modelos"

                st.markdown(f"""
                <div class="consensus">
                  <div class="consensus-label">// Consenso · Probabilidad media</div>
                  <div class="consensus-value" style="color:{cc}">{ct}</div>
                  <div class="consensus-sub">P(&gt;50K) promedio = <span>{avg_prob:.1%}</span></div>
                </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#3a4155;">
          <div style="font-size:3rem;margin-bottom:1rem;opacity:.35">◈</div>
          <div style="font-family:'Space Mono',monospace;font-size:.75rem;letter-spacing:.2em;
                      text-transform:uppercase;color:#4a536a;">
            Ingresa los datos y pulsa<br>PREDECIR INGRESO
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Info pipelines ─────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">// Arquitectura de Preprocesamiento</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="font-size:.82rem;line-height:1.7;">
      <div class="card-title">Pipelines activos</div>
      <div class="pipe-row">
        <span class="chip chip-cyan">pipeline_log</span>
        <span class="pipe-name">Keras &amp; PyTorch</span>
        <div class="pipe-desc">Regionalización → Agrupación edu/ocu → log1p(capital) → StandardScaler + OHE</div>
      </div>
      <div class="pipe-row">
        <span class="chip chip-amber">nativo (category)</span>
        <span class="pipe-name">XGBoost</span>
        <div class="pipe-desc">Regionalización → CastCategorías (dtype category) — 12 features, sin escalado ni OHE</div>
      </div>
    </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:1rem;border-top:1px solid #1e2130;margin-top:1rem;">
  <span style="font-family:'Space Mono',monospace;font-size:.65rem;letter-spacing:.15em;color:#2e3449;">
    CENSUS INCOME PREDICTOR · UCI ADULT DATASET · KERAS · PYTORCH · XGBOOST
  </span>
</div>""", unsafe_allow_html=True)