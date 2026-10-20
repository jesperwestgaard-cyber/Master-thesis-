#!/usr/bin/env python3
"""Create thesis figures for the raw-KO baseline evaluation."""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
X_PATH = BASE / "ml_feature_matrix.csv"
Y_PATH = BASE / "ml_target_matrix.csv"
METRICS_PATH = BASE / "ml_baseline_results" / "raw_ko_baseline_metrics.csv"
OUTPUT_DIR = BASE / "Figurer" / "baseline_thesis"
KO_PATTERN = re.compile(r"K\d{5}\Z")
TRAIT_ORDER = ["Chemo", "Litho", "Auto", "Hetero", "Photo", "Organo"]
MODEL_ORDER = ["LinearSVM", "LogisticRegression", "XGBoost", "RandomForest"]
MODEL_LABELS = {
    "LinearSVM": "Linear SVM",
    "LogisticRegression": "Logistic regression",
    "XGBoost": "XGBoost",
    "RandomForest": "Random forest",
}
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#666666",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D9DEE3",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{stem}.{extension}",
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def load_inputs():
    x_frame = pd.read_csv(X_PATH)
    y_frame = pd.read_csv(Y_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    if x_frame["sample"].duplicated().any() or y_frame["sample"].duplicated().any():
        raise ValueError("Sample IDs must be unique in X and Y matrices.")
    if y_frame["sample"].isin(x_frame["sample"]).sum() != len(y_frame):
        raise ValueError("Some Y-matrix samples are missing from the X matrix.")

    # Plot only the exact samples used by the baseline model comparison.
    x_aligned = x_frame.set_index("sample").loc[y_frame["sample"]]
    ko_columns = [column for column in x_frame.columns if KO_PATTERN.fullmatch(column)]
    traits = [column for column in y_frame.columns if column not in {"sample", "assembly"}]
    if traits != TRAIT_ORDER:
        raise ValueError(f"Unexpected target columns: {traits}")
    return x_aligned, y_frame, metrics, ko_columns


def plot_trait_prevalence(y_frame: pd.DataFrame) -> None:
    positives = y_frame[TRAIT_ORDER].sum().reindex(TRAIT_ORDER)
    sample_count = len(y_frame)
    percentages = positives / sample_count * 100
    y_positions = np.arange(len(TRAIT_ORDER))

    fig, ax = plt.subplots(figsize=(8.0, 4.4), layout="constrained")
    bars = ax.barh(y_positions, percentages, color=COLORS["blue"], height=0.66)
    ax.set_yticks(y_positions, labels=TRAIT_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, max(percentages) * 1.22)
    ax.set_xlabel("Samples with trait (%)")
    ax.set_title("Prevalence of nutritional traits", loc="left", weight="bold", pad=12)
    ax.text(
        0,
        1.02,
        f"n = {sample_count} labeled samples; multilabel classification",
        transform=ax.transAxes,
        color=COLORS["gray"],
        fontsize=9,
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for bar, count, percentage in zip(bars, positives, percentages):
        ax.text(
            bar.get_width() + max(percentages) * 0.018,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count)}  ({percentage:.1f}%)",
            va="center",
            fontsize=9,
        )
    save_figure(fig, "01_trait_prevalence")


def plot_ko_prevalence(x_aligned: pd.DataFrame, y_frame: pd.DataFrame, ko_columns) -> None:
    # Count a KO once per genome assembly even when several BacDive strains map
    # to that assembly, matching the grouping used for model validation.
    assembly_features = x_aligned.groupby("assembly", sort=False)[ko_columns].first()
    prevalence = assembly_features.sum(axis=0).to_numpy()
    bin_edges = [0, 1, 2, 5, 10, 50, 100, len(assembly_features) + 1]
    bin_labels = ["0", "1", "2–4", "5–9", "10–49", "50–99", "100+"]
    counts = [
        int(((prevalence >= bin_edges[index]) & (prevalence < bin_edges[index + 1])).sum())
        for index in range(len(bin_labels))
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.4), layout="constrained")
    positions = np.arange(len(bin_labels))
    bars = ax.bar(positions, counts, color=COLORS["sky"], width=0.68)
    ax.set_xticks(positions, labels=bin_labels)
    ax.set_xlabel("Assemblies containing KO")
    ax.set_ylabel("Number of KO features")
    ax.set_title("Prevalence of KO features", loc="left", weight="bold", pad=12)
    ax.text(
        0,
        1.02,
        f"{len(ko_columns):,} KO features across {len(assembly_features)} assemblies",
        transform=ax.transAxes,
        color=COLORS["gray"],
        fontsize=9,
    )
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", visible=True)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylim(0, max(counts) * 1.14)
    save_figure(fig, "02_ko_prevalence")


def plot_macro_metrics(metrics: pd.DataFrame) -> None:
    macro = metrics.loc[metrics["trait"] == "MACRO_AVERAGE"].copy()
    macro["model"] = pd.Categorical(macro["model"], MODEL_ORDER, ordered=True)
    macro = macro.sort_values("model")
    metric_names = ["precision", "accuracy", "mcc"]
    metric_labels = ["Precision", "Accuracy", "MCC"]
    metric_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    y_positions = np.arange(len(macro))
    bar_height = 0.22

    fig, ax = plt.subplots(figsize=(9.0, 4.8), layout="constrained")
    for metric_index, (metric, label, color) in enumerate(
        zip(metric_names, metric_labels, metric_colors)
    ):
        offsets = (metric_index - 1) * bar_height
        values = macro[metric].to_numpy()
        bars = ax.barh(
            y_positions + offsets,
            values,
            height=bar_height * 0.9,
            label=label,
            color=color,
        )
        for bar, value in zip(bars, values):
            ax.text(
                value + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=8,
            )
    ax.set_yticks(y_positions, labels=[MODEL_LABELS[str(model)] for model in macro["model"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Macro-average score across six traits")
    ax.set_title("Raw-KO baseline model comparison", loc="left", weight="bold", pad=12)
    ax.legend(frameon=False, ncols=3, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    save_figure(fig, "03_baseline_macro_metrics")


def plot_trait_mcc(metrics: pd.DataFrame) -> None:
    trait_metrics = metrics.loc[metrics["trait"].isin(TRAIT_ORDER)].copy()
    score_table = trait_metrics.pivot(index="model", columns="trait", values="mcc")
    score_table = score_table.reindex(index=MODEL_ORDER, columns=TRAIT_ORDER)

    fig, ax = plt.subplots(figsize=(9.0, 4.6), layout="constrained")
    image = ax.imshow(score_table.to_numpy(), cmap="YlGnBu", vmin=0, vmax=0.9, aspect="auto")
    ax.set_xticks(np.arange(len(TRAIT_ORDER)), labels=TRAIT_ORDER)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), labels=[MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_title("Matthews correlation by nutritional trait", loc="left", weight="bold", pad=12)
    for row_index in range(len(MODEL_ORDER)):
        for column_index in range(len(TRAIT_ORDER)):
            value = score_table.iloc[row_index, column_index]
            text_color = "white" if value >= 0.58 else "#1E2933"
            ax.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88, pad=0.03)
    colorbar.set_label("MCC")
    ax.text(
        0,
        -0.2,
        "Photo has 11 positive samples; interpret its score cautiously.",
        transform=ax.transAxes,
        color=COLORS["gray"],
        fontsize=9,
    )
    save_figure(fig, "04_trait_mcc_heatmap")


def main() -> None:
    configure_style()
    x_aligned, y_frame, metrics, ko_columns = load_inputs()
    plot_trait_prevalence(y_frame)
    plot_ko_prevalence(x_aligned, y_frame, ko_columns)
    plot_macro_metrics(metrics)
    plot_trait_mcc(metrics)
    print(f"Generated four baseline figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()