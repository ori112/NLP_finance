"""FinBERT inference pipeline with sliding window chunking for long texts.

Model: ProsusAI/finbert (HuggingFace Hub)
  - Pre-trained on financial text (10-K filings, financial news)
  - Three output classes: positive, negative, neutral
  - 512-token BERT limit handled via sliding_window_chunks()

Caching: After the first run, scores are saved to data/outputs/finbert_scores.parquet.
On subsequent calls, the cache is loaded directly — re-inference for 200 transcripts
takes 1-2 hours on CPU, so caching is essential.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

from src.models.chunker import aggregate_chunk_scores, sliding_window_chunks

logger = logging.getLogger(__name__)

MODEL_ID = "ProsusAI/finbert"
_SCORE_CACHE = Path("data/outputs/finbert_scores.parquet")
_LABEL_MAP = {"positive": "positive", "negative": "negative", "neutral": "neutral"}


def load_finbert_pipeline(device: int = -1):
    """Load FinBERT as a HuggingFace text-classification pipeline.

    Args:
        device: -1 for CPU; 0 for first CUDA GPU; auto-detected if None.
                On machines without a GPU, always use -1.

    Returns:
        HuggingFace pipeline object ready for inference.
    """
    if device == -1 and torch.cuda.is_available():
        logger.info("CUDA GPU detected — switching to device=0 for faster inference.")
        device = 0

    logger.info("Loading FinBERT from %s on device=%d …", MODEL_ID, device)
    pipe = hf_pipeline(
        "text-classification",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        device=device,
        top_k=None,         # return scores for ALL three classes
        truncation=True,    # safety — chunker should prevent hitting the limit
        max_length=512,
    )
    logger.info("FinBERT loaded.")
    return pipe


def predict_sentiment(
    text: str,
    finbert_pipeline,
    max_tokens: int = 512,
    stride: int = 128,
    aggregation: str = "mean",
) -> dict[str, float]:
    """Run FinBERT on a (potentially long) text using sliding window chunking.

    Args:
        text: Clean segment text (prepared remarks or Q&A).
        finbert_pipeline: Loaded HuggingFace pipeline from load_finbert_pipeline().
        max_tokens: Max tokens per chunk (BERT limit = 512).
        stride: Sliding window stride in tokens.
        aggregation: Score aggregation strategy ('mean', 'majority').

    Returns:
        Dict with keys: positive, negative, neutral (float probabilities),
        label (str), n_chunks (int).
    """
    tokenizer = finbert_pipeline.tokenizer
    chunks = sliding_window_chunks(text, tokenizer, max_tokens=max_tokens, stride=stride)

    chunk_scores: list[dict[str, float]] = []
    for chunk in chunks:
        raw = finbert_pipeline(chunk)
        # raw is a list of lists when top_k=None: [[{label, score}, ...]]
        if isinstance(raw[0], list):
            scores_list = raw[0]
        else:
            scores_list = raw
        chunk_score = {
            _LABEL_MAP.get(item["label"].lower(), item["label"].lower()): item["score"]
            for item in scores_list
        }
        chunk_scores.append(chunk_score)

    result = aggregate_chunk_scores(chunk_scores, strategy=aggregation)
    result["n_chunks"] = len(chunks)
    return result


def score_dataframe(
    df: pd.DataFrame,
    text_col: str,
    finbert_pipeline,
    prefix: str = "finbert",
    batch_size: int = 1,
    aggregation: str = "mean",
    cache_path: Path = _SCORE_CACHE,
) -> pd.DataFrame:
    """Apply FinBERT inference to every row in df[text_col].

    Results are cached to a Parquet file. If the cache exists and covers
    the same (ticker, date, text_col) combination, it is returned directly
    without running inference again.

    Added columns (with given prefix):
      {prefix}_positive, {prefix}_negative, {prefix}_neutral,
      {prefix}_label, {prefix}_n_chunks

    Args:
        df: DataFrame with at least text_col, ticker, date columns.
        text_col: Column containing the text to score.
        finbert_pipeline: Loaded FinBERT pipeline.
        prefix: Column name prefix for output columns.
        batch_size: Inference batch size (use 8+ on GPU, 1 on CPU).
        aggregation: Chunk aggregation strategy.
        cache_path: Path to the Parquet cache file.

    Returns:
        DataFrame with FinBERT score columns appended.
    """
    cache_col = f"{prefix}_label"

    # Load from cache if available and complete
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if cache_col in cached.columns and len(cached) == len(df):
            logger.info("Loading FinBERT scores from cache: %s", cache_path)
            # Merge cached scores back onto df by position
            score_cols = [c for c in cached.columns if c.startswith(prefix)]
            return pd.concat(
                [df.reset_index(drop=True), cached[score_cols].reset_index(drop=True)],
                axis=1,
            )

    logger.info("Running FinBERT inference on %d rows (text_col=%s) …", len(df), text_col)
    results = []
    for text in tqdm(df[text_col], desc=f"FinBERT ({prefix})"):
        result = predict_sentiment(text, finbert_pipeline, aggregation=aggregation)
        results.append(result)

    scores_df = pd.DataFrame(results)
    scores_df.columns = [f"{prefix}_{c}" for c in scores_df.columns]
    output = pd.concat([df.reset_index(drop=True), scores_df], axis=1)

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(cache_path, index=False)
    logger.info("FinBERT scores cached to %s", cache_path)

    return output
