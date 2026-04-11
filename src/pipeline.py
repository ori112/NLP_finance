"""Full analysis pipeline orchestrator.

Invoked by: python main.py --mode analyze [--model lm|finbert|both]

Steps:
  1. Load preprocessed segments.parquet (run preprocessing first if missing).
  2. Score with LM Lexicon and/or FinBERT.
  3. Evaluate all conditions and compute metrics.
  4. Extract and export failure cases (≥10 required).
  5. Generate all visualizations.
  6. Save results_summary.csv.
"""

import logging
from pathlib import Path

import pandas as pd

from src.evaluation.error_analysis import export_failure_cases, extract_failure_cases
from src.evaluation.metrics import compare_models
from src.evaluation.visualizations import (
    plot_model_comparison_f1,
    plot_pearson_heatmap,
    plot_sentiment_distribution,
    plot_sentiment_vs_returns,
)
from src.models.lm_lexicon import score_dataframe as lm_score

logger = logging.getLogger(__name__)

_SEGMENTS_PATH = Path("data/processed/segments.parquet")
_RESULTS_PATH = Path("data/outputs/results_summary.csv")
_FAILURES_PATH = Path("data/outputs/failure_cases.csv")


def _load_segments() -> pd.DataFrame:
    if not _SEGMENTS_PATH.exists():
        raise FileNotFoundError(
            f"Segments file not found at {_SEGMENTS_PATH}. "
            "Run preprocessing first: python main.py --mode analyze will attempt it, "
            "or run src/processing/pipeline.py directly."
        )
    return pd.read_parquet(_SEGMENTS_PATH)


def run_analysis(model: str = "both") -> None:
    """Run the full sentiment analysis and evaluation pipeline.

    Args:
        model: Which model(s) to run — 'lm', 'finbert', or 'both'.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # ---- Load data -----------------------------------------------------------
    logger.info("Loading segments from %s …", _SEGMENTS_PATH)
    df = _load_segments()
    logger.info("Loaded %d rows (%d with successful splits).",
                len(df), df["split_successful"].sum() if "split_successful" in df.columns else len(df))

    # ---- LM Lexicon scoring --------------------------------------------------
    if model in ("lm", "both"):
        logger.info("Scoring with LM Lexicon …")
        df = lm_score(df, token_col="prepared_tokens", prefix="pr_lm")
        df = lm_score(df, token_col="qa_tokens", prefix="qa_lm")

    # ---- FinBERT scoring -----------------------------------------------------
    if model in ("finbert", "both"):
        from src.models.finbert import load_finbert_pipeline
        from src.models.finbert import score_dataframe as finbert_score

        logger.info("Loading FinBERT pipeline …")
        pipe = load_finbert_pipeline()

        logger.info("Scoring prepared remarks with FinBERT …")
        df = finbert_score(df, text_col="prepared_text", finbert_pipeline=pipe, prefix="pr_finbert")

        logger.info("Scoring Q&A sections with FinBERT …")
        df = finbert_score(df, text_col="qa_text", finbert_pipeline=pipe, prefix="qa_finbert",
                           cache_path=Path("data/outputs/finbert_scores_qa.parquet"))

    # ---- Evaluation ----------------------------------------------------------
    logger.info("Computing evaluation metrics …")
    results_df = compare_models(df, ground_truth_col="label_1d")
    logger.info("\n%s", results_df.to_string(index=False))

    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(_RESULTS_PATH, index=False)
    logger.info("Results saved to %s", _RESULTS_PATH)

    # ---- Error analysis (includes heuristic-split failures) ------------------
    logger.info("Extracting failure cases …")
    pred_col = "pr_lm_label" if "pr_lm_label" in df.columns else "pr_finbert_label"
    try:
        failures = extract_failure_cases(df, model_label_col=pred_col)
        export_failure_cases(failures, output_path=_FAILURES_PATH)
    except ValueError as exc:
        logger.error("Failure case export error: %s", exc)

    # ---- Visualizations ------------------------------------------------------
    logger.info("Generating visualizations …")

    if "pr_lm_net_sentiment" in df.columns:
        plot_sentiment_vs_returns(
            df, sentiment_col="pr_lm_net_sentiment",
            title="LM Lexicon (Prepared Remarks) vs. Market Return",
            output_path=Path("data/outputs/figures/lm_pr_vs_returns.png"),
        )
        plot_sentiment_vs_returns(
            df, sentiment_col="qa_lm_net_sentiment",
            title="LM Lexicon (Q&A) vs. Market Return",
            output_path=Path("data/outputs/figures/lm_qa_vs_returns.png"),
        )

    plot_model_comparison_f1(results_df)
    plot_sentiment_distribution(df)
    plot_pearson_heatmap(results_df)

    logger.info("Analysis complete. Outputs in data/outputs/")
