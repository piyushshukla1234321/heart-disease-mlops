"""Exploratory data analysis figures.

Run from the repository root:
    python -m cardiorisk.eda        # writes PNGs to reports/figures/eda_*.png

Each plot_* function returns a matplotlib Figure so the EDA notebook can show
the same figures inline.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from cardiorisk import config as C

log = logging.getLogger(__name__)
PALETTE = {0: "#4C9F70", 1: "#C8553D"}       # green = no disease, red = disease
TARGET_NAMES = {0: "No disease", 1: "Disease"}


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["diagnosis"] = out[C.TARGET].map(TARGET_NAMES)
    return out


def plot_class_balance(df: pd.DataFrame) -> plt.Figure:
    counts = df[C.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar([TARGET_NAMES[i] for i in counts.index], counts.values,
                  color=[PALETTE[i] for i in counts.index])
    for bar, n in zip(bars, counts.values):
        ax.annotate(f"{n}  ({n / counts.sum():.0%})", (bar.get_x() + bar.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Patients")
    ax.set_title("Class balance of the binary target")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    return fig


def plot_missing_values(df: pd.DataFrame) -> plt.Figure:
    miss = df[C.FEATURES].isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(miss.index, miss.values, color="#5B7DB1")
    ax.set_title("Missing values per feature after cleaning (imputed inside the pipeline)")
    ax.set_ylabel("Missing cells")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_numeric_distributions(df: pd.DataFrame) -> plt.Figure:
    data = _labelled(df)
    cols = C.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(data=data, x=col, hue="diagnosis", kde=True, stat="density",
                     common_norm=False, element="step", ax=ax,
                     palette={TARGET_NAMES[k]: v for k, v in PALETTE.items()})
        ax.set_title(C.FEATURE_DESCRIPTIONS[col], fontsize=10)
        ax.set_xlabel("")
    fig.suptitle("Numeric feature distributions by diagnosis (each class normalised separately)")
    fig.tight_layout()
    return fig


def plot_disease_rate_by_category(df: pd.DataFrame) -> plt.Figure:
    """For each coded feature: share of patients with disease in every category."""
    cols = C.CATEGORICAL_FEATURES + C.BINARY_FEATURES
    overall = df[C.TARGET].mean()
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    for ax, col in zip(axes.ravel(), cols):
        grp = df.dropna(subset=[col]).groupby(col)[C.TARGET].agg(["mean", "size"])
        labels = [C.CODE_LABELS[col].get(int(k), str(k)) for k in grp.index]
        ax.bar(labels, grp["mean"], color="#8E6C8A")
        for i, (rate, n) in enumerate(zip(grp["mean"], grp["size"])):
            ax.text(i, rate + 0.02, f"{rate:.0%}\nn={n}", ha="center", fontsize=8)
        ax.axhline(overall, ls="--", color="grey", lw=1)
        ax.set_ylim(0, 1.15)
        ax.set_title(C.FEATURE_DESCRIPTIONS[col], fontsize=10)
        ax.tick_params(axis="x", rotation=25, labelsize=8)
    axes.ravel()[-1].axis("off")
    fig.suptitle(f"Disease rate per category (dashed line = overall rate {overall:.0%})")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    # Spearman (rank) correlation: robust to skew and fine for ordinal codes like ca.
    cols = C.NUMERIC_FEATURES + C.BINARY_FEATURES + [C.TARGET]
    corr = df[cols].corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="vlag", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Spearman rho"})
    ax.set_title("Spearman correlation (numeric + binary features and target)")
    fig.tight_layout()
    return fig


def plot_outlier_boxplots(df: pd.DataFrame) -> plt.Figure:
    data = _labelled(df).melt(id_vars="diagnosis", value_vars=C.NUMERIC_FEATURES)
    # z-score each feature so all fit on one axis
    data["z"] = data.groupby("variable")["value"].transform(lambda s: (s - s.mean()) / s.std())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.boxplot(data=data, x="variable", y="z", hue="diagnosis", ax=ax,
                palette={TARGET_NAMES[k]: v for k, v in PALETTE.items()})
    ax.set_xlabel("")
    ax.set_ylabel("z-score")
    ax.set_title("Standardised numeric features by diagnosis (points = potential outliers)")
    fig.tight_layout()
    return fig


def key_findings(df: pd.DataFrame) -> dict:
    """Computed facts for the report - nothing here is hard-coded."""
    cols = C.NUMERIC_FEATURES + C.BINARY_FEATURES
    rho = df[cols + [C.TARGET]].corr(method="spearman")[C.TARGET].drop(C.TARGET)
    top = rho.reindex(rho.abs().sort_values(ascending=False).index).head(5)
    return {
        "n_patients": int(len(df)),
        "positive_rate": round(float(df[C.TARGET].mean()), 3),
        "rows_with_any_missing": int(df[C.FEATURES].isna().any(axis=1).sum()),
        "top_spearman_with_target": {k: round(float(v), 3) for k, v in top.items()},
    }


FIGURES = {
    "eda_class_balance": plot_class_balance,
    "eda_missing_values": plot_missing_values,
    "eda_numeric_distributions": plot_numeric_distributions,
    "eda_disease_rate_by_category": plot_disease_rate_by_category,
    "eda_correlation_heatmap": plot_correlation_heatmap,
    "eda_outlier_boxplots": plot_outlier_boxplots,
}


def generate_all(df: pd.DataFrame, out_dir: Path = C.FIGURES_DIR) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in FIGURES.items():
        fig = fn(df)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        log.info("Wrote %s", path)
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from cardiorisk.data import load_clean
    df = load_clean()
    generate_all(df)
    for key, value in key_findings(df).items():
        log.info("  %-26s %s", key, value)


if __name__ == "__main__":
    main()
