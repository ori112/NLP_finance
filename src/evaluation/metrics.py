"""Evaluation metrics: classification (accuracy, F1) and correlation (Pearson)."""

import logging

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report, f1_score

logger = logging.getLogger(__name__)

_LABELS = ["up", "down"]


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] = _LABELS,
) -> dict[str, float]:
    """Compute accuracy and F1 scores for a set of predictions.

    Binary stock-direction task: labels are "up" and "down".

    Args:
        y_true: Ground-truth labels (e.g. from market-adjusted returns).
        y_pred: Model-predicted labels.
        labels: Ordered class labels for per-class F1 reporting.

    Returns:
        Dict with keys: accuracy, macro_f1, weighted_f1, up_f1, down_f1.
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "up_f1": round(per_class[0], 4),
        "down_f1": round(per_class[1], 4),
    }


def compute_pearson_correlation(
    sentiment_scores: list[float],
    market_returns: list[float],
) -> tuple[float, float]:
    """Compute Pearson correlation between continuous sentiment scores and returns.

    Args:
        sentiment_scores: Continuous scores (e.g. net_sentiment or finbert_positive
                          minus finbert_negative). Must match length of market_returns.
        market_returns: Market-adjusted stock returns (e.g. market_adj_1d).

    Returns:
        Tuple of (pearson_r, p_value). p_value < 0.05 indicates statistical significance.
    """
    if len(sentiment_scores) < 3:
        logger.warning("Pearson requires at least 3 data points; got %d.", len(sentiment_scores))
        return (0.0, 1.0)

    r, p = pearsonr(sentiment_scores, market_returns)
    return (round(float(r), 4), round(float(p), 4))


def compare_models(
    df: pd.DataFrame,
    ground_truth_col: str = "label_1d",
) -> pd.DataFrame:
    """Build the 4-way comparison table across models and segments.

    Evaluates each of these conditions against the ground truth label:
      - PR_LM    : Prepared Remarks scored by LM Lexicon
      - QA_LM    : Q&A section scored by LM Lexicon
      - PR_FINBERT: Prepared Remarks scored by FinBERT
      - QA_FINBERT: Q&A section scored by FinBERT

    Only rows where split_successful=True are included in evaluation.

    Args:
        df: Master DataFrame with all model scores and ground truth.
        ground_truth_col: Column name of the ground-truth label.

    Returns:
        Summary DataFrame with one row per condition and metric columns.
    """
    eval_df = df[df["split_successful"]].dropna(subset=[ground_truth_col])
    y_true = eval_df[ground_truth_col].tolist()

    conditions = {
        "PR_LM": "pr_lm_label",
        "QA_LM": "qa_lm_label",
        "PR_FINBERT": "pr_finbert_label",
        "QA_FINBERT": "qa_finbert_label",
    }

    rows = []
    for condition, pred_col in conditions.items():
        if pred_col not in eval_df.columns:
            logger.warning("Column %s not found — skipping condition %s.", pred_col, condition)
            continue

        y_pred = eval_df[pred_col].tolist()
        metrics = compute_classification_metrics(y_true, y_pred)

        # Pearson correlation using continuous scores
        pearson_map = {
            "PR_LM": ("pr_lm_net_sentiment", "market_adj_1d"),
            "QA_LM": ("qa_lm_net_sentiment", "market_adj_1d"),
            "PR_FINBERT": ("pr_finbert_positive", "market_adj_1d"),
            "QA_FINBERT": ("qa_finbert_positive", "market_adj_1d"),
        }
        score_col, return_col = pearson_map[condition]
        valid = eval_df[[score_col, return_col]].dropna()
        if len(valid) >= 3:
            r1d, p1d = compute_pearson_correlation(
                valid[score_col].tolist(), valid[return_col].tolist()
            )
        else:
            r1d, p1d = 0.0, 1.0

        # Also compute 3d Pearson
        return_col_3d = return_col.replace("1d", "3d")
        valid3 = eval_df[[score_col, return_col_3d]].dropna()
        if len(valid3) >= 3:
            r3d, p3d = compute_pearson_correlation(
                valid3[score_col].tolist(), valid3[return_col_3d].tolist()
            )
        else:
            r3d, p3d = 0.0, 1.0

        rows.append({
            "condition": condition,
            "n_samples": len(eval_df),
            **metrics,
            "pearson_r_1d": r1d,
            "pearson_p_1d": p1d,
            "pearson_r_3d": r3d,
            "pearson_p_3d": p3d,
        })

    return pd.DataFrame(rows)
