"""
EDA Interactivo - Adult Census Income Dataset
Examen Parcial 2026-1 | Estadística Informática
Ejecutar con: streamlit run 01_eda_parcial.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EDA · Adult Census Income",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta consistente
PAL = {
    "blue":   "#378ADD",
    "coral":  "#D85A30",
    "teal":   "#5DCAA5",
    "amber":  "#EF9F27",
    "pink":   "#D4537E",
    "lgray":  "#B4B2A9",
    "lblue":  "#85B7EB",
    "green":  "#639922",
    "purple": "#7F77DD",
}
CLASS_COLORS = {"<=50K": PAL["blue"], ">50K": PAL["coral"]}

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
    "axes.labelsize":     11,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# ─────────────────────────────────────────────
# CARGA DE DATOS (con caché)
# ─────────────────────────────────────────────
COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income",
]

EDU_ORDER = [
    "Preschool", "1st-4th", "5th-6th", "7th-8th", "9th",
    "10th", "11th", "12th", "HS-grad", "Some-college",
    "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters",
    "Prof-school", "Doctorate",
]

@st.cache_data(show_spinner="Cargando dataset...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, header=None, names=COLS,
        sep=r",\s*", engine="python", na_values="?",
    )
    df["income"] = df["income"].str.strip()
    df["target"] = (df["income"] == ">50K").astype(int)

    df["age_group"] = pd.cut(
        df["age"],
        bins=[16, 29, 39, 49, 59, 100],
        labels=["17-29", "30-39", "40-49", "50-59", "60+"],
    )
    df["edu_group"] = pd.cut(
        df["education_num"],
        bins=[0, 8, 9, 11, 13, 16],
        labels=["Básica", "Secundaria", "Técnico", "Universitario", "Posgrado"],
    )
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    df["capital_net"] = df["capital_gain"] - df["capital_loss"]
    df["log_capital_gain"] = np.log1p(df["capital_gain"])
    df["log_fnlwgt"] = np.log(df["fnlwgt"])
    df["is_married"] = df["marital_status"].isin(
        ["Married-civ-spouse", "Married-AF-spouse"]
    ).astype(int)
    df["hours_bin"] = pd.cut(
        df["hours_per_week"],
        bins=[0, 34, 39, 40, 45, 50, 99],
        labels=["<35h", "35-39h", "40h", "41-45h", "46-50h", ">50h"],
    )
    df["is_college"] = (df["education_num"] >= 13).astype(int)
    return df


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📊 EDA · Adult Census")
    st.caption("Examen Parcial 2026-1")
    st.divider()

    data_source = st.radio(
        "Fuente de datos",
        ["Archivo local (adult.data)", "Subir archivo"],
    )

    if data_source == "Subir archivo":
        uploaded = st.file_uploader("Sube adult.data", type=["data", "csv"])
        if uploaded:
            df = load_data(uploaded)
        else:
            st.warning("Sube el archivo para continuar.")
            st.stop()
    else:
        default_path = Path(__file__).parent / "adult.data"
        if not default_path.exists():
            st.error(
                "No se encontró `adult.data` en la carpeta del script. "
                "Cambia a 'Subir archivo' o coloca el archivo junto a este script."
            )
            st.stop()
        df = load_data(str(default_path))

    st.success(f"✅ {len(df):,} instancias cargadas")

    seccion = st.radio(
        "Sección",
        [
            "1 · Visión General",
            "2 · Distribuciones",
            "3 · Variable Objetivo",
            "4 · Análisis Multivariado",
            "5 · Capital Gain / Loss",
            "6 · Valores Nulos",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Desarrollado para el examen parcial de Redes Neuronales · 2026-1")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fig_to_st(fig, caption: str = ""):
    st.pyplot(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    plt.close(fig)


def rate_by(df, col):
    return (
        df.groupby(col)["target"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "rate", "count": "n"})
        .assign(rate_pct=lambda x: x["rate"] * 100)
        .sort_values("rate_pct", ascending=False)
    )


# ─────────────────────────────────────────────
# SECCIÓN 1 · VISIÓN GENERAL
# ─────────────────────────────────────────────
if seccion == "1 · Visión General":
    st.header("Visión General del Dataset")
    st.markdown(
        "**Dataset:** Adult Census Income · 1994 US Census  \n"
        "**Tarea:** Clasificación binaria — predecir si ingreso anual **>50K USD**"
    )

    n_pos = df["target"].sum()
    n_neg = len(df) - n_pos
    nulls = df[["workclass", "occupation", "native_country"]].isna().sum().sum()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Instancias", f"{len(df):,}")
    k2.metric("Variables", "14")
    k3.metric("Clase >50K", f"{n_pos/len(df)*100:.1f}%")
    k4.metric("Desbalance", f"{n_neg/n_pos:.1f}×")
    k5.metric("Valores nulos", f"{nulls:,}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de la variable objetivo")
        st.caption("Desbalance 3:1 — usar ROC-AUC y F1, no Accuracy")
        counts = df["income"].value_counts()
        pcts   = counts / counts.sum() * 100

        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.barh(
            counts.index, pcts.values,
            color=[CLASS_COLORS.get(i, PAL["lgray"]) for i in counts.index],
            height=0.5, edgecolor="none",
        )
        for bar, pct in zip(bars, pcts.values):
            ax.text(
                bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=11, fontweight="bold",
            )
        ax.set_xlim(0, 90)
        ax.set_xlabel("Porcentaje (%)")
        ax.set_title("Clases de ingreso", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

    with col2:
        st.subheader("Tasa >50K por cuartil de variables numéricas")
        st.caption("Comparación de poder predictivo entre variables continuas")

        num_vars = ["age", "education_num", "hours_per_week"]
        labels_vars = ["Edad", "Educación\n(años)", "Horas/\nsemana"]
        quartile_rates = {}
        for v in num_vars:
            qcut = pd.qcut(df[v], q=4, labels=False, duplicates="drop")
            rates = (
                df.groupby(qcut, observed=True)["target"].mean() * 100
            ).values
            full_rates = np.full(4, np.nan)
            full_rates[:len(rates)] = rates
            quartile_rates[v] = full_rates

        x = np.arange(len(num_vars))
        w = 0.2
        q_colors = [PAL["lblue"], PAL["teal"], PAL["amber"], PAL["coral"]]

        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for i, (q_label, color) in enumerate(
            zip(["Q1", "Q2", "Q3", "Q4"], q_colors)
        ):
            rates = [quartile_rates[v][i] for v in num_vars]
            ax.bar(x + i * w, rates, width=w, label=q_label, color=color,
                   edgecolor="none")

        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels(labels_vars, fontsize=10)
        ax.set_ylabel("Tasa >50K (%)")
        ax.set_title("Poder predictivo por cuartil", fontsize=12, fontweight="bold")
        ax.legend(title="Cuartil", fontsize=9, title_fontsize=9,
                  loc="upper left", framealpha=0.7)
        fig.tight_layout()
        fig_to_st(fig)

    st.subheader("Estadísticas descriptivas — variables numéricas")
    num_cols = ["age", "fnlwgt", "education_num", "capital_gain",
                "capital_loss", "hours_per_week"]
    st.dataframe(
        df[num_cols].describe().T.style.format("{:.2f}").background_gradient(
            cmap="Blues", subset=["mean", "std"]
        ),
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# SECCIÓN 2 · DISTRIBUCIONES UNIVARIADAS
# ─────────────────────────────────────────────
elif seccion == "2 · Distribuciones":
    st.header("Distribuciones Univariadas")

    tab1, tab2, tab3 = st.tabs(
        ["Variables numéricas", "Variables categóricas", "Educación detalle"]
    )

    with tab1:
        st.subheader("Distribución de edad — separada por clase de ingreso")
        st.caption(
            "La distribución >50K está desplazada ~8 años a la derecha. "
            "La mediana de >50K es ~44 vs ~35 de ≤50K."
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        bins = range(17, 92, 3)
        for income_class, color in CLASS_COLORS.items():
            subset = df[df["income"] == income_class]["age"]
            ax.hist(subset, bins=bins, alpha=0.6, color=color,
                    label=income_class, edgecolor="none", density=True)
        ax.set_xlabel("Edad")
        ax.set_ylabel("Densidad")
        ax.set_title("Histograma de edad (densidad)", fontweight="bold")
        ax.legend()

        ax = axes[1]
        data_box = [
            df[df["income"] == c]["age"].values
            for c in ["<=50K", ">50K"]
        ]
        bp = ax.boxplot(
            data_box, labels=["≤50K", ">50K"], patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )
        colors_box = [PAL["blue"], PAL["coral"]]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_ylabel("Edad")
        ax.set_title("Boxplot de edad por clase", fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig, "Izq: densidades superpuestas | Der: cuartiles y outliers")

        st.divider()
        st.subheader("Horas trabajadas por semana")
        st.caption(
            "Pico dominante en 40h (jornada estándar). "
            "Los que ganan >50K tienen mayor dispersión hacia jornadas largas."
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax = axes[0]
        for income_class, color in CLASS_COLORS.items():
            subset = df[df["income"] == income_class]["hours_per_week"]
            ax.hist(subset, bins=range(1, 100, 2), alpha=0.65, color=color,
                    label=income_class, density=True, edgecolor="none")
        ax.axvline(40, color="black", linestyle="--", alpha=0.5, linewidth=1,
                   label="40h (moda)")
        ax.set_xlabel("Horas por semana")
        ax.set_ylabel("Densidad")
        ax.set_title("Distribución de horas trabajadas", fontweight="bold")
        ax.legend(fontsize=9)

        ax = axes[1]
        violin_data = [
            df[df["income"] == c]["hours_per_week"].values
            for c in ["<=50K", ">50K"]
        ]
        vp = ax.violinplot(violin_data, showmedians=True, showextrema=True)
        for pc, color in zip(vp["bodies"], [PAL["blue"], PAL["coral"]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["≤50K", ">50K"])
        ax.set_ylabel("Horas por semana")
        ax.set_title("Violin plot — horas/semana", fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Workclass")
            wc_counts = df["workclass"].value_counts().dropna()
            fig, ax = plt.subplots(figsize=(5.5, 4))
            colors_wc = [PAL["blue"], PAL["teal"], PAL["teal"],
                         PAL["amber"], PAL["amber"], PAL["green"], PAL["lgray"]]
            bars = ax.barh(wc_counts.index, wc_counts.values,
                           color=colors_wc[:len(wc_counts)], edgecolor="none")
            for bar in bars:
                ax.text(
                    bar.get_width() + 100,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():,.0f}",
                    va="center", fontsize=9,
                )
            ax.set_xlabel("Frecuencia")
            ax.set_title("Tipo de empleador (workclass)", fontweight="bold")
            fig.tight_layout()
            fig_to_st(fig)

        with col2:
            st.subheader("Raza y Sexo")
            fig, axes = plt.subplots(2, 1, figsize=(5, 5.5))

            race_counts = df["race"].value_counts()
            axes[0].bar(range(len(race_counts)), race_counts.values,
                        color=[PAL["blue"], PAL["coral"], PAL["teal"],
                               PAL["amber"], PAL["purple"]][:len(race_counts)],
                        edgecolor="none")
            axes[0].set_xticks(range(len(race_counts)))
            axes[0].set_xticklabels(race_counts.index, rotation=20,
                                    ha="right", fontsize=9)
            axes[0].set_ylabel("Frecuencia")
            axes[0].set_title("Distribución por raza", fontweight="bold")

            sex_counts = df["sex"].value_counts()
            wedges, texts, autotexts = axes[1].pie(
                sex_counts.values,
                labels=sex_counts.index,
                autopct="%1.1f%%",
                colors=[PAL["blue"], PAL["pink"]],
                startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2),
            )
            for at in autotexts:
                at.set_fontsize(10)
                at.set_fontweight("bold")
            axes[1].set_title("Distribución por sexo", fontweight="bold")
            fig.tight_layout()
            fig_to_st(fig)

        st.subheader("Estado civil")
        mc_counts = df["marital_status"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.bar(range(len(mc_counts)), mc_counts.values,
                      color=[PAL["coral"] if "Married" in x else PAL["lblue"]
                             for x in mc_counts.index],
                      edgecolor="none")
        ax.set_xticks(range(len(mc_counts)))
        ax.set_xticklabels(mc_counts.index, rotation=25, ha="right")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 80,
                    f"{bar.get_height():,}", ha="center", fontsize=9)
        ax.set_ylabel("Frecuencia")
        ax.set_title("Estado civil — los casados son mayoría", fontweight="bold")
        fig.tight_layout()
        fig_to_st(
            fig,
            "🟥 Casado/a   🟦 Otro estado — el matrimonio es fuertemente correlacionado con >50K",
        )

    with tab3:
        st.subheader("Perfil completo del nivel educativo")
        edu_counts = df["education"].value_counts()
        edu_present = [e for e in EDU_ORDER if e in edu_counts.index]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        vals = [edu_counts.get(e, 0) for e in edu_present]
        bar_colors = [PAL["coral"] if e in ["Bachelors", "Masters", "Doctorate",
                      "Prof-school"] else PAL["lblue"] for e in edu_present]
        ax.barh(edu_present, vals, color=bar_colors, edgecolor="none")
        ax.set_xlabel("Frecuencia")
        ax.set_title("Frecuencia por nivel educativo", fontweight="bold")
        ax.invert_yaxis()

        ax = axes[1]
        edu_rate = rate_by(df, "education").loc[
            [e for e in edu_present if e in rate_by(df, "education").index]
        ]
        edu_rate_ordered = edu_rate.reindex([e for e in edu_present
                                             if e in edu_rate.index])
        bar_colors2 = [
            PAL["coral"] if v > 40 else PAL["amber"] if v > 20 else PAL["lblue"]
            for v in edu_rate_ordered["rate_pct"].values
        ]
        ax.barh(
            edu_rate_ordered.index,
            edu_rate_ordered["rate_pct"].values,
            color=bar_colors2, edgecolor="none",
        )
        ax.axvline(24.1, color="gray", linestyle="--", linewidth=1,
                   alpha=0.6, label="Tasa global (24.1%)")
        ax.set_xlabel("Tasa >50K (%)")
        ax.set_title("Tasa >50K por nivel educativo", fontweight="bold")
        ax.legend(fontsize=9)
        ax.invert_yaxis()

        fig.tight_layout()
        fig_to_st(
            fig,
            "Izq: frecuencia absoluta | Der: tasa de ingreso >50K — "
            "relación casi monótona con educación_num",
        )


# ─────────────────────────────────────────────
# SECCIÓN 3 · VARIABLE OBJETIVO
# ─────────────────────────────────────────────
elif seccion == "3 · Variable Objetivo":
    st.header("Análisis de la Variable Objetivo (>50K)")
    st.info(
        "🔍 **Variables más discriminantes:** estado civil casado (44.7%), "
        "educación doctoral/maestría (>55%), ocupación exec-managerial (48.4%) "
        "y capital gain > 0 (~75%)."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Ocupación y Educación", "Demografía", "Horas y Trabajo"]
    )

    with tab1:
        st.subheader("Tasa >50K por ocupación")
        occ_rate = rate_by(df, "occupation").dropna()
        fig, ax = plt.subplots(figsize=(9, 5))
        colors_occ = [
            PAL["coral"] if v > 35 else PAL["amber"] if v > 20 else PAL["lblue"]
            for v in occ_rate["rate_pct"].values
        ]
        bars = ax.barh(occ_rate.index, occ_rate["rate_pct"].values,
                       color=colors_occ, edgecolor="none")
        ax.axvline(24.1, color="black", linestyle="--", linewidth=1,
                   alpha=0.5, label="Tasa global 24.1%")
        for bar, n in zip(bars, occ_rate["n"].values):
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%  (n={n:,})",
                va="center", fontsize=9,
            )
        ax.set_xlim(0, 60)
        ax.set_xlabel("Tasa de ingreso >50K (%)")
        ax.set_title("Exec-managerial y Prof-specialty lideran con ~45-48%",
                     fontweight="bold")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig_to_st(fig)

        st.subheader("Tasa >50K por estado civil")
        mc_rate = rate_by(df, "marital_status")
        fig, ax = plt.subplots(figsize=(9, 3.5))
        bar_colors_mc = [
            PAL["coral"] if v > 30 else PAL["amber"] if v > 10 else PAL["lblue"]
            for v in mc_rate["rate_pct"].values
        ]
        bars = ax.barh(mc_rate.index, mc_rate["rate_pct"].values,
                       color=bar_colors_mc, edgecolor="none")
        ax.axvline(24.1, color="black", linestyle="--", linewidth=1,
                   alpha=0.5, label="Tasa global 24.1%")
        for bar in bars:
            ax.text(
                bar.get_width() + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1f}%", va="center", fontsize=10,
            )
        ax.set_xlim(0, 55)
        ax.set_xlabel("Tasa de ingreso >50K (%)")
        ax.set_title(
            "Casados: 44.7% vs Solteros: 4.6% — la mayor brecha del dataset",
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig_to_st(
            fig,
            "⚠️ Posible confusión con edad y sexo — Married-civ-spouse incluye "
            "principalmente hombres de mediana edad.",
        )

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tasa >50K por raza y sexo")
            fig, ax = plt.subplots(figsize=(6, 4))
            races = df["race"].value_counts().index[:5]
            x = np.arange(len(races))
            w = 0.35
            for i, (sex, color) in enumerate(
                [("Male", PAL["blue"]), ("Female", PAL["pink"])]
            ):
                rates_sex = []
                for r in races:
                    sub = df[(df["race"] == r) & (df["sex"] == sex)]
                    rates_sex.append(sub["target"].mean() * 100 if len(sub) > 0 else 0)
                ax.bar(x + i * w, rates_sex, width=w, label=sex, color=color,
                       edgecolor="none")

            ax.set_xticks(x + w / 2)
            ax.set_xticklabels(races, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Tasa >50K (%)")
            ax.set_title("Brecha de género consistente\nen todas las razas",
                         fontweight="bold")
            ax.legend()
            ax.axhline(24.1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            fig.tight_layout()
            fig_to_st(fig)

        with col2:
            st.subheader("Tasa >50K por grupo de edad")
            age_rate = (
                df.groupby("age_group", observed=True)["target"]
                .agg(["mean", "count"])
                .assign(rate_pct=lambda x: x["mean"] * 100)
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            bar_colors_age = [
                PAL["lblue"], PAL["teal"], PAL["amber"], PAL["coral"], PAL["pink"]
            ]
            bars = ax.bar(
                age_rate.index.astype(str), age_rate["rate_pct"].values,
                color=bar_colors_age, edgecolor="none",
            )
            ax2 = ax.twinx()
            ax2.plot(
                range(len(age_rate)), age_rate["count"].values,
                color="gray", marker="o", linewidth=1.5, markersize=4,
                alpha=0.6, label="n instancias",
            )
            ax2.set_ylabel("Frecuencia", color="gray", fontsize=10)
            ax2.tick_params(axis="y", labelcolor="gray")
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{bar.get_height():.1f}%",
                    ha="center", fontsize=9, fontweight="bold",
                )
            ax.set_xlabel("Grupo de edad")
            ax.set_ylabel("Tasa >50K (%)")
            ax.set_title("Pico en 40-49 años\n(eje der. = frecuencia)",
                         fontweight="bold")
            ax.set_ylim(0, 45)
            fig.tight_layout()
            fig_to_st(fig)

        st.subheader("Probabilidad >50K según edad — curva suavizada")
        age_smooth = (
            df.groupby("age")["target"].mean().reset_index()
            .rename(columns={"target": "rate"})
        )
        age_smooth["rate_smooth"] = (
            age_smooth["rate"].rolling(window=5, center=True, min_periods=1).mean()
        )
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.fill_between(age_smooth["age"], age_smooth["rate_smooth"] * 100,
                        alpha=0.25, color=PAL["coral"])
        ax.plot(age_smooth["age"], age_smooth["rate_smooth"] * 100,
                color=PAL["coral"], linewidth=2, label="P(>50K | edad)")
        ax.axhline(24.1, color="gray", linestyle="--", linewidth=1,
                   alpha=0.5, label="Tasa global 24.1%")
        ax.set_xlabel("Edad")
        ax.set_ylabel("Tasa >50K (%)")
        ax.set_title(
            "La probabilidad de ganar >50K crece hasta ~45 años y luego decrece",
            fontweight="bold",
        )
        ax.legend()
        fig.tight_layout()
        fig_to_st(fig, "Ventana de suavizado: 5 años. Datos reales por año de edad.")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tasa >50K por workclass")
            wc_rate = rate_by(df, "workclass").dropna()
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            bar_colors_wc = [
                PAL["coral"] if v > 30 else PAL["amber"] if v > 20 else PAL["lblue"]
                for v in wc_rate["rate_pct"].values
            ]
            bars = ax.barh(wc_rate.index, wc_rate["rate_pct"].values,
                           color=bar_colors_wc, edgecolor="none")
            ax.axvline(24.1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            for bar in bars:
                ax.text(
                    bar.get_width() + 0.3,
                    bar.get_y() + bar.get_height() / 2,
                    f"{bar.get_width():.1f}%", va="center", fontsize=9,
                )
            ax.set_xlim(0, 50)
            ax.set_xlabel("Tasa >50K (%)")
            ax.set_title("Workclass vs. tasa de ingreso", fontweight="bold")
            fig.tight_layout()
            fig_to_st(fig)

        with col2:
            st.subheader("Tasa >50K por horas trabajadas")
            hours_rate = (
                df.groupby("hours_bin", observed=True)["target"]
                .mean() * 100
            )
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            hrs_colors = [PAL["lblue"], PAL["teal"], PAL["amber"],
                          PAL["amber"], PAL["coral"], PAL["coral"]]
            bars = ax.bar(hours_rate.index.astype(str), hours_rate.values,
                          color=hrs_colors[:len(hours_rate)], edgecolor="none")
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}%",
                    ha="center", fontsize=9, fontweight="bold",
                )
            ax.axhline(24.1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_xlabel("Horas por semana")
            ax.set_ylabel("Tasa >50K (%)")
            ax.set_title("Más horas → mayor tasa, pero no lineal", fontweight="bold")
            fig.tight_layout()
            fig_to_st(fig)


# ─────────────────────────────────────────────
# SECCIÓN 4 · ANÁLISIS MULTIVARIADO
# ─────────────────────────────────────────────
elif seccion == "4 · Análisis Multivariado":
    st.header("Análisis Multivariado")
    st.info(
        "💡 **Interacción clave:** Edad × Educación × Estado civil explican "
        "la mayor parte de la varianza predictiva."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Heatmap Edad × Educación", "Workclass × Educación", "Correlación numérica"]
    )

    with tab1:
        st.subheader("Heatmap: Tasa >50K — Edad (filas) × Educación (columnas)")
        st.caption(
            "Intensidad = % de personas que ganan >50K. "
            "Combinaciones con n < 30 se marcan en gris."
        )
        pivot = (
            df.groupby(["age_group", "edu_group"], observed=True)["target"]
            .agg(["mean", "count"])
            .reset_index()
            .pivot(index="age_group", columns="edu_group", values="mean") * 100
        )
        pivot_n = (
            df.groupby(["age_group", "edu_group"], observed=True)["target"]
            .count()
            .reset_index()
            .pivot(index="age_group", columns="edu_group", values="target")
        )
        mask = pivot_n < 30
        pivot_plot = pivot.copy()
        pivot_plot[mask] = np.nan

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            pivot_plot,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="white",
            ax=ax,
            cbar_kws={"label": "Tasa >50K (%)"},
            mask=mask,
            vmin=0, vmax=80,
        )
        for i in range(pivot_plot.shape[0]):
            for j in range(pivot_plot.shape[1]):
                if mask.iloc[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j, i), 1, 1, fill=True, color="#DDDDDD",
                    ))
                    ax.text(j + 0.5, i + 0.5, "n<30",
                            ha="center", va="center", fontsize=8, color="#999")
        ax.set_title(
            "Tasa de ingreso >50K (%) por grupo de edad y nivel educativo",
            fontweight="bold", fontsize=13,
        )
        ax.set_xlabel("Nivel educativo", fontsize=11)
        ax.set_ylabel("Grupo de edad", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig_to_st(fig, "Gris = menos de 30 observaciones (no confiable estadísticamente)")

        st.markdown("""
        **Lecturas clave del heatmap:**
        - La celda **Posgrado × 40-49 años** es la de mayor tasa (~68%)
        - La combinación **Básica × cualquier edad** rara vez supera 10%
        - La edad amplifica el efecto de la educación: un universitario de 40-49 tiene ~2× más probabilidad que uno de 17-29
        """)

    with tab2:
        st.subheader("Efecto de la educación dentro de cada tipo de empleo")
        st.caption(
            "Los autónomos incorporados (Self-emp-inc) con educación universitaria "
            "tienen la tasa más alta del dataset (~72%)."
        )
        result = []
        for wc in df["workclass"].dropna().unique():
            for edu_flag, edu_label in [(0, "No universitario"), (1, "Universitario+")]:
                sub = df[(df["workclass"] == wc) & (df["is_college"] == edu_flag)]
                if len(sub) >= 20:
                    result.append({
                        "workclass": wc,
                        "educacion": edu_label,
                        "rate": sub["target"].mean() * 100,
                        "n": len(sub),
                    })
        result_df = pd.DataFrame(result)

        wcs = result_df["workclass"].unique()
        x = np.arange(len(wcs))
        w = 0.35
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, (edu, color) in enumerate(
            [("No universitario", PAL["lblue"]), ("Universitario+", PAL["coral"])]
        ):
            sub_r = result_df[result_df["educacion"] == edu].set_index("workclass")
            vals = [sub_r.loc[wc, "rate"] if wc in sub_r.index else 0 for wc in wcs]
            bars = ax.bar(x + i * w, vals, width=w, label=edu, color=color,
                          edgecolor="none")
            for bar in bars:
                if bar.get_height() > 1:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{bar.get_height():.0f}%",
                        ha="center", fontsize=8,
                    )

        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(wcs, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Tasa >50K (%)")
        ax.set_title(
            "La educación universitaria amplifica la tasa en todos los sectores",
            fontweight="bold",
        )
        ax.legend()
        ax.axhline(24.1, color="gray", linestyle="--", linewidth=1, alpha=0.4)
        fig.tight_layout()
        fig_to_st(fig)

        st.subheader("Edad media por ocupación y clase de ingreso")
        st.caption(
            "Dentro de cada ocupación, quienes ganan >50K son sistemáticamente "
            "~5-7 años mayores."
        )
        occ_age = (
            df.groupby(["occupation", "income"], observed=True)["age"]
            .mean()
            .reset_index()
            .dropna()
        )
        occs = occ_age["occupation"].unique()
        x = np.arange(len(occs))
        w = 0.35
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for i, (inc, color) in enumerate(
            [("<=50K", PAL["lblue"]), (">50K", PAL["coral"])]
        ):
            sub_r = occ_age[occ_age["income"] == inc].set_index("occupation")
            vals = [sub_r.loc[o, "age"] if o in sub_r.index else 0 for o in occs]
            ax.bar(x + i * w, vals, width=w, label=inc, color=color,
                   edgecolor="none")
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(occs, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Edad media (años)")
        ax.set_title("Edad media por ocupación y clase de ingreso", fontweight="bold")
        ax.legend()
        ax.set_ylim(30, 52)
        fig.tight_layout()
        fig_to_st(fig)

    with tab3:
        st.subheader("Matriz de correlación — variables numéricas + target")
        num_cols = [
            "age", "education_num", "hours_per_week",
            "capital_gain", "capital_loss", "log_capital_gain",
            "capital_net", "target",
        ]
        corr = df[num_cols].corr()
        mask_tri = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            corr, mask=mask_tri, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, linewidths=0.5,
            linecolor="white", ax=ax, vmin=-0.6, vmax=0.6,
            cbar_kws={"label": "Correlación de Pearson"},
        )
        ax.set_title("Correlación Pearson — incluye variables derivadas",
                     fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
        fig.tight_layout()
        fig_to_st(
            fig,
            "log_capital_gain y education_num son las variables numéricas "
            "con mayor correlación con target (~0.33 y ~0.34 respectivamente).",
        )


# ─────────────────────────────────────────────
# SECCIÓN 5 · CAPITAL GAIN / LOSS
# ─────────────────────────────────────────────
elif seccion == "5 · Capital Gain / Loss":
    st.header("Capital Gain y Capital Loss")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Capital gain > 0", f"{df['has_capital_gain'].mean()*100:.1f}%",
              "de la muestra")
    k2.metric("Capital loss > 0", f"{df['has_capital_loss'].mean()*100:.1f}%",
              "de la muestra")
    tasa_gain = df[df["has_capital_gain"] == 1]["target"].mean()
    tasa_nogain = df[df["has_capital_gain"] == 0]["target"].mean()
    k3.metric("Tasa >50K con gain", f"{tasa_gain*100:.1f}%",
              f"+{(tasa_gain-tasa_nogain)*100:.1f}pp vs sin gain")
    k4.metric("Tasa >50K sin gain", f"{tasa_nogain*100:.1f}%")

    st.warning(
        "⚡ **Capital gain es la variable más discriminante del dataset.** "
        "Candidata obligatoria a ingeniería de variables: "
        "`has_capital_gain` (flag), `log1p(capital_gain)`, `capital_net`."
    )

    tab1, tab2 = st.tabs(["Distribución y discriminación", "Interacciones"])

    with tab1:
        st.subheader("Distribución de capital gain — escala logarítmica (solo >0)")
        pos_gain = df[df["capital_gain"] > 0].copy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        ax = axes[0]
        bins = np.logspace(np.log10(1), np.log10(100000), 25)
        for inc, color in CLASS_COLORS.items():
            sub = pos_gain[pos_gain["income"] == inc]["capital_gain"]
            ax.hist(sub, bins=bins, alpha=0.65, color=color, label=inc,
                    edgecolor="none", density=True)
        ax.set_xscale("log")
        ax.set_xlabel("Capital gain (log scale, USD)")
        ax.set_ylabel("Densidad")
        ax.set_title("Distribución capital gain > 0\npor clase de ingreso",
                     fontweight="bold")
        ax.legend()

        ax = axes[1]
        data_vl = [
            pos_gain[pos_gain["income"] == c]["log_capital_gain"].values
            for c in ["<=50K", ">50K"]
        ]
        vp = ax.violinplot(data_vl, showmedians=True)
        for pc, color in zip(vp["bodies"], [PAL["blue"], PAL["coral"]]):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["≤50K", ">50K"])
        ax.set_ylabel("log(1 + capital_gain)")
        ax.set_title("Violin — log capital gain por clase", fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

        st.subheader("Tasa >50K según presencia de capital gain/loss")
        groups = {
            "Sin gain\nni loss": df[(df["has_capital_gain"] == 0) & (df["has_capital_loss"] == 0)],
            "Solo\ncapital gain": df[(df["has_capital_gain"] == 1) & (df["has_capital_loss"] == 0)],
            "Solo\ncapital loss": df[(df["has_capital_gain"] == 0) & (df["has_capital_loss"] == 1)],
            "Gain\ny loss": df[(df["has_capital_gain"] == 1) & (df["has_capital_loss"] == 1)],
        }
        labels_g = list(groups.keys())
        rates_g = [g["target"].mean() * 100 for g in groups.values()]
        counts_g = [len(g) for g in groups.values()]

        fig, ax = plt.subplots(figsize=(9, 3.5))
        bar_colors_g = [PAL["lblue"], PAL["coral"], PAL["teal"], PAL["amber"]]
        bars = ax.bar(labels_g, rates_g, color=bar_colors_g, edgecolor="none",
                      width=0.5)
        for bar, cnt in zip(bars, counts_g):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%\n(n={cnt:,})",
                ha="center", fontsize=9, fontweight="bold",
            )
        ax.axhline(24.1, color="gray", linestyle="--", linewidth=1,
                   alpha=0.5, label="Tasa global 24.1%")
        ax.set_ylabel("Tasa >50K (%)")
        ax.set_title("Presencia de transacciones de capital vs. clase de ingreso",
                     fontweight="bold")
        ax.set_ylim(0, 85)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig_to_st(fig)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Capital gain por nivel educativo")
            edu_gain = (
                df[df["has_capital_gain"] == 1]
                .groupby("edu_group", observed=True)["capital_gain"]
                .median()
            )
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.bar(edu_gain.index.astype(str), edu_gain.values,
                   color=[PAL["lblue"], PAL["teal"], PAL["amber"],
                          PAL["coral"], PAL["purple"]],
                   edgecolor="none")
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
            )
            ax.set_xlabel("Nivel educativo")
            ax.set_ylabel("Mediana capital gain (USD)")
            ax.set_title("Mediana de gain solo entre\nquienes tienen gain > 0",
                         fontweight="bold")
            fig.tight_layout()
            fig_to_st(fig)

        with col2:
            st.subheader("Tasa >50K: horas × capital gain")
            result_ch = []
            for hb in df["hours_bin"].dropna().cat.categories:
                for cg, cg_label in [(0, "Sin gain"), (1, "Con gain")]:
                    sub = df[(df["hours_bin"] == hb) & (df["has_capital_gain"] == cg)]
                    if len(sub) >= 15:
                        result_ch.append({
                            "hours_bin": str(hb),
                            "capital": cg_label,
                            "rate": sub["target"].mean() * 100,
                        })
            ch_df = pd.DataFrame(result_ch)
            hrs_u = ch_df["hours_bin"].unique()
            x = np.arange(len(hrs_u))
            w = 0.35
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            for i, (cg, color) in enumerate(
                [("Sin gain", PAL["blue"]), ("Con gain", PAL["amber"])]
            ):
                sub_c = ch_df[ch_df["capital"] == cg].set_index("hours_bin")
                vals = [sub_c.loc[h, "rate"] if h in sub_c.index else 0 for h in hrs_u]
                ax.bar(x + i * w, vals, width=w, label=cg, color=color, edgecolor="none")
            ax.set_xticks(x + w / 2)
            ax.set_xticklabels(hrs_u, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Tasa >50K (%)")
            ax.set_title("Horas × Capital gain\n(tasa de ingreso alto)",
                         fontweight="bold")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig_to_st(fig)


# ─────────────────────────────────────────────
# SECCIÓN 6 · VALORES NULOS
# ─────────────────────────────────────────────
elif seccion == "6 · Valores Nulos":
    st.header("Análisis de Valores Nulos")

    null_counts = df[COLS].isna().sum()
    null_pct = null_counts / len(df) * 100
    null_df = pd.DataFrame({
        "Variable":   null_counts.index,
        "N nulos":    null_counts.values,
        "% nulos":    null_pct.values.round(2),
    }).query("`N nulos` > 0")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Resumen de nulos")
        st.dataframe(null_df.style.format({"% nulos": "{:.2f}%"}),
                     use_container_width=True, hide_index=True)
        st.markdown("""
        **Estrategias de imputación recomendadas:**
        - `workclass`: moda o categoría `"Unknown"`
        - `occupation`: moda condicional por `workclass`
        - `native_country`: moda (`United-States`) o `"Unknown"`

        ⚠️ Aplicar imputación **dentro del pipeline** sklearn para evitar data leakage.
        """)

    with col2:
        st.subheader("¿Los nulos son aleatorios? (MCAR test visual)")
        st.caption(
            "Si la tasa de >50K es similar entre nulos y no-nulos, "
            "los datos faltantes son probablemente MCAR."
        )
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for ax, col_null in zip(axes, ["workclass", "occupation", "native_country"]):
            mask_null = df[col_null].isna()
            rates_null = [
                df[~mask_null]["target"].mean() * 100,
                df[mask_null]["target"].mean() * 100,
            ]
            ax.bar(
                ["No nulo", "Nulo"],
                rates_null,
                color=[PAL["blue"], PAL["coral"]],
                edgecolor="none", width=0.5,
            )
            for j, v in enumerate(rates_null):
                ax.text(j, v + 0.3, f"{v:.1f}%", ha="center",
                        fontsize=10, fontweight="bold")
            ax.axhline(24.1, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_title(col_null, fontweight="bold")
            ax.set_ylabel("Tasa >50K (%)" if ax == axes[0] else "")
            ax.set_ylim(0, 35)
        fig.suptitle(
            "Tasa >50K en filas con y sin valor nulo — si difieren, "
            "los nulos son informativos (MAR/MNAR)",
            fontsize=10, y=1.02,
        )
        fig.tight_layout()
        fig_to_st(fig)

        st.subheader("Distribución de edad en filas con nulos vs sin nulos")
        fig, ax = plt.subplots(figsize=(9, 3))
        has_any_null = df[COLS].isna().any(axis=1)
        ax.hist(df[~has_any_null]["age"], bins=30, alpha=0.6,
                color=PAL["blue"], label="Sin nulos", density=True)
        ax.hist(df[has_any_null]["age"], bins=30, alpha=0.6,
                color=PAL["coral"], label="Con nulos", density=True)
        ax.set_xlabel("Edad")
        ax.set_ylabel("Densidad")
        ax.legend()
        ax.set_title(
            "Distribución de edad: filas completas vs filas con algún nulo",
            fontweight="bold",
        )
        fig.tight_layout()
        fig_to_st(
            fig,
            "Si las distribuciones son similares → nulos aleatorios (MCAR). "
            "Diferencias → MAR o MNAR, imputación más cuidadosa.",
        )