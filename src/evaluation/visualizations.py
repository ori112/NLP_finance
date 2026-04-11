"""Visualization functions for the evaluation report.

All plots are saved at 300 DPI to data/outputs/figures/ for inclusion
in the 5-page report and presentation slides.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

_FIG_DIR = Path("data/outputs/figures")
sns.set_theme(style="whitegrid", palette="muted")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def plot_sentiment_vs_returns(
    df: pd.DataFrame,
    sentiment_col: str,
    return_col: str = "market_adj_1d",
    label_col: str = "label_1d",
    title: str = "Sentiment Score vs. Market-Adjusted Return",
    output_path: Path = _FIG_DIR / "sentiment_vs_returns.png",
) -> None:
    """Scatter plot of continuous sentiment score vs. market-adjusted return.

    Each point is coloured by ground-truth label (positive/neutral/negative).
    The Pearson r and p-value are annotated on the plot.

    Args:
        df: DataFrame with sentiment score, return, and label columns.
        sentiment_col: Column with continuous sentiment values (e.g. lm_net_sentiment).
        return_col: Column with continuous return values.
        label_col: Column with categorical ground-truth label for colouring.
        title: Plot title.
        output_path: Where to save the PNG.
    """
    plot_df = df[[sentiment_col, return_col, label_col]].dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

    for label, group in plot_df.groupby(label_col):
        ax.scatter(
            group[sentiment_col],
            group[return_col],
            label=label,
            color=palette.get(label, "grey"),
            alpha=0.6,
            edgecolors="white",
            linewidths=0.4,
            s=40,
        )

    r, p = pearsonr(plot_df[sentiment_col], plot_df[return_col])
    ax.annotate(
        f"Pearson r = {r:.3f}  (p = {p:.3f})",
        xy=(0.05, 0.93),
        xycoords="axes fraction",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(sentiment_col.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(return_col.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(title="Market Direction", fontsize=9)

    _save(fig, output_path)


def plot_model_comparison_f1(
    results_df: pd.DataFrame,
    output_path: Path = _FIG_DIR / "model_comparison_f1.png",
) -> None:
    """Grouped bar chart comparing macro and weighted F1 across all 4 conditions.

    Args:
        results_df: Output of evaluation.metrics.compare_models().
        output_path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    conditions = results_df["condition"].tolist()
    x = np.arange(len(conditions))
    width = 0.35

    ax.bar(x - width / 2, results_df["macro_f1"], width, label="Macro F1", color="#3498db")
    ax.bar(x + width / 2, results_df["weighted_f1"], width, label="Weighted F1", color="#e67e22")

    for i, (macro, weighted) in enumerate(
        zip(results_df["macro_f1"], results_df["weighted_f1"])
    ):
        ax.text(i - width / 2, macro + 0.005, f"{macro:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, weighted + 0.005, f"{weighted:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison: F1 Scores Across Conditions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    _save(fig, output_path)


def plot_sentiment_distribution(
    df: pd.DataFrame,
    label_cols: list[str] | None = None,
    output_path: Path = _FIG_DIR / "sentiment_distribution.png",
) -> None:
    """Stacked bar chart showing sentiment label distribution for each model/segment.

    Helps identify class imbalance that could inflate accuracy metrics.

    Args:
        df: Master DataFrame with model label columns.
        label_cols: List of label column names to plot. Defaults to all *_label cols.
        output_path: Where to save the PNG.
    """
    if label_cols is None:
        label_cols = [c for c in df.columns if c.endswith("_label") and c != "label_1d"]

    counts = {}
    for col in label_cols:
        vc = df[col].value_counts()
        counts[col] = {lbl: vc.get(lbl, 0) for lbl in ["positive", "neutral", "negative"]}

    count_df = pd.DataFrame(counts).T

    fig, ax = plt.subplots(figsize=(10, 5))
    count_df.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color={"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"},
        edgecolor="white",
        width=0.6,
    )

    ax.set_xlabel("Model / Segment", fontsize=11)
    ax.set_ylabel("Number of Transcripts", fontsize=11)
    ax.set_title("Sentiment Label Distribution by Model and Segment", fontsize=13, fontweight="bold")
    ax.legend(title="Sentiment", fontsize=9, loc="upper right")
    ax.tick_params(axis="x", rotation=30)

    _save(fig, output_path)


def plot_pearson_heatmap(
    results_df: pd.DataFrame,
    output_path: Path = _FIG_DIR / "pearson_heatmap.png",
) -> None:
    """Heatmap of Pearson r values: conditions × return windows (1d, 3d).

    Immediately shows which model + segment combination correlates best
    with actual stock price movement.

    Args:
        results_df: Output of evaluation.metrics.compare_models().
        output_path: Where to save the PNG.
    """
    pivot = results_df.set_index("condition")[["pearson_r_1d", "pearson_r_3d"]]
    pivot.columns = ["1-Day Return", "3-Day Return"]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 11},
    )
    ax.set_title("Pearson Correlation: Sentiment vs. Market Returns", fontsize=13, fontweight="bold")
    ax.set_ylabel("")

    _save(fig, output_path)
