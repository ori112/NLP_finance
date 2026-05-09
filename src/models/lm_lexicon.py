"""Loughran-McDonald financial sentiment lexicon scorer.

The LM Master Dictionary is distributed as a CSV from the University of Notre Dame:
  https://sraf.nd.edu/loughranmcdonald-master-dictionary/

Download `Loughran-McDonald_MasterDictionary_1993-2023.csv` and place it at:
  data/raw/lm_dictionary/Loughran-McDonald_MasterDictionary_1993-2023.csv

Dictionary columns used (nonzero value = word belongs to category):
  Word, Positive, Negative, Uncertainty, Litigious, Constraining
"""

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LM_CSV_PATH = Path(
    "data/raw/lm_dictionary/Loughran-McDonald_MasterDictionary_1993-2025.csv"
)

_CATEGORIES = ["Positive", "Negative", "Uncertainty", "Litigious", "Constraining"]


@lru_cache(maxsize=1)
def load_lm_dictionary(csv_path: Path = LM_CSV_PATH) -> dict[str, set[str]]:
    """Load the LM Master Dictionary CSV into category word-sets.

    The result is cached with lru_cache so the CSV is read exactly once
    per Python process, regardless of how many times this function is called.

    Args:
        csv_path: Path to the LM Master Dictionary CSV file.

    Returns:
        Dict mapping lowercase category name → set of uppercase word strings.

    Raises:
        FileNotFoundError: If the CSV file does not exist at csv_path.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"LM dictionary not found at {csv_path}.\n"
            "Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/\n"
            "and place at: data/raw/lm_dictionary/"
        )

    df = pd.read_csv(csv_path, usecols=["Word"] + _CATEGORIES, low_memory=False)
    df["Word"] = df["Word"].str.upper().str.strip()

    result: dict[str, set[str]] = {}
    for col in _CATEGORIES:
        result[col.lower()] = set(df.loc[df[col] != 0, "Word"])

    logger.info(
        "LM dictionary loaded: %s",
        {k: len(v) for k, v in result.items()},
    )
    return result


def compute_lm_score(tokens: list[str]) -> dict[str, float]:
    """Compute normalised LM sentiment scores for a list of tokens.

    Scores are normalised by token count so transcripts of different lengths
    are directly comparable.

    Formula for each category c:
        score_c = count(tokens ∩ LM[c]) / max(len(tokens), 1)

    net_sentiment = positive_ratio - negative_ratio

    Args:
        tokens: List of UPPERCASE tokens (output of tokenize_for_lexicon()).

    Returns:
        Dict with keys:
          positive_ratio, negative_ratio, net_sentiment,
          uncertainty_ratio, litigious_ratio, constraining_ratio, n_tokens.
    """
    if len(tokens) == 0:
        return {
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "net_sentiment": 0.0,
            "uncertainty_ratio": 0.0,
            "litigious_ratio": 0.0,
            "constraining_ratio": 0.0,
            "n_tokens": 0,
        }

    lm = load_lm_dictionary()
    n = len(tokens)
    token_set = tokens  # keep as list for counting (duplicates matter)

    pos = sum(1 for t in token_set if t in lm["positive"]) / n
    neg = sum(1 for t in token_set if t in lm["negative"]) / n
    unc = sum(1 for t in token_set if t in lm["uncertainty"]) / n
    lit = sum(1 for t in token_set if t in lm["litigious"]) / n
    con = sum(1 for t in token_set if t in lm["constraining"]) / n

    return {
        "positive_ratio": round(pos, 6),
        "negative_ratio": round(neg, 6),
        "net_sentiment": round(pos - neg, 6),
        "uncertainty_ratio": round(unc, 6),
        "litigious_ratio": round(lit, 6),
        "constraining_ratio": round(con, 6),
        "n_tokens": n,
    }


def classify_sentiment(scores: dict[str, float], threshold: float = 0.0) -> str:
    """Map LM net_sentiment to a binary direction prediction (up/down).

    The proposal frames the task as binary stock-direction classification, so
    threshold = 0 is the natural sign split: more positive financial words
    than negative ones predicts "up", otherwise "down".

    Args:
        scores: Output of compute_lm_score().
        threshold: Boundary above which net_sentiment predicts "up".

    Returns:
        "up" if net_sentiment > threshold else "down".
    """
    return "up" if scores["net_sentiment"] > threshold else "down"


def score_dataframe(
    df: pd.DataFrame,
    token_col: str,
    prefix: str = "lm",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Apply LM scoring to every row in df[token_col].

    Adds columns:
      {prefix}_positive_ratio, {prefix}_negative_ratio, {prefix}_net_sentiment,
      {prefix}_uncertainty_ratio, {prefix}_litigious_ratio,
      {prefix}_constraining_ratio, {prefix}_n_tokens, {prefix}_label

    Call twice — once with token_col='prepared_tokens' and prefix='pr_lm',
    once with token_col='qa_tokens' and prefix='qa_lm' — to score both segments.

    Args:
        df: DataFrame containing a column of token lists.
        token_col: Column name with lists of uppercase tokens.
        prefix: Column name prefix for the output score columns.
        threshold: Sentiment classification threshold.

    Returns:
        DataFrame with LM score columns appended (original df is not mutated).
    """
    scores = df[token_col].apply(compute_lm_score)
    scores_df = pd.DataFrame(scores.tolist())
    scores_df.columns = [f"{prefix}_{c}" for c in scores_df.columns]
    scores_df[f"{prefix}_label"] = scores_df[f"{prefix}_net_sentiment"].apply(
        lambda net: classify_sentiment({"net_sentiment": net}, threshold)
    )
    return pd.concat([df.reset_index(drop=True), scores_df], axis=1)
