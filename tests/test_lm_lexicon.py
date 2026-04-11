"""Tests for the Loughran-McDonald lexicon scorer.

These tests run without the actual LM dictionary CSV by patching load_lm_dictionary
with a minimal in-memory fixture. The CSV-loading test is marked slow/integration.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.models.lm_lexicon import classify_sentiment, compute_lm_score, score_dataframe

# ---------------------------------------------------------------------------
# Minimal fixture dictionary (uppercase, mimics LM structure)
# ---------------------------------------------------------------------------

_MOCK_LM: dict[str, set[str]] = {
    "positive": {"GROWTH", "PROFIT", "OPPORTUNITY", "REVENUE", "STRONG", "RECORD"},
    "negative": {"RISK", "LOSS", "DECLINE", "UNCERTAINTY", "WEAK", "CONCERN"},
    "uncertainty": {"UNCERTAIN", "UNCLEAR", "APPROXIMATELY", "ROUGHLY"},
    "litigious": {"LITIGATION", "LAWSUIT", "PLAINTIFF", "ALLEGED"},
    "constraining": {"MUST", "SHALL", "REQUIRED", "OBLIGATED"},
}


def _patch_lm(func):
    """Decorator: patch load_lm_dictionary with _MOCK_LM for the test."""
    return patch("src.models.lm_lexicon.load_lm_dictionary", return_value=_MOCK_LM)(func)


# ---------------------------------------------------------------------------
# compute_lm_score
# ---------------------------------------------------------------------------

@_patch_lm
def test_compute_score_positive_text(*_) -> None:
    tokens = ["GROWTH", "PROFIT", "OPPORTUNITY", "REVENUE", "QUARTER"]
    scores = compute_lm_score(tokens)
    assert scores["net_sentiment"] > 0
    assert scores["positive_ratio"] > 0
    assert scores["negative_ratio"] == 0.0


@_patch_lm
def test_compute_score_negative_text(*_) -> None:
    tokens = ["RISK", "LOSS", "DECLINE", "CONCERN", "QUARTER"]
    scores = compute_lm_score(tokens)
    assert scores["net_sentiment"] < 0
    assert scores["negative_ratio"] > 0
    assert scores["positive_ratio"] == 0.0


@_patch_lm
def test_compute_score_neutral_text(*_) -> None:
    tokens = ["QUARTER", "FISCAL", "YEAR", "MANAGEMENT", "COMPANY"]
    scores = compute_lm_score(tokens)
    assert scores["net_sentiment"] == 0.0


@_patch_lm
def test_compute_score_empty_tokens(*_) -> None:
    scores = compute_lm_score([])
    assert scores["net_sentiment"] == 0.0
    assert scores["n_tokens"] == 0


@_patch_lm
def test_compute_score_returns_all_expected_keys(*_) -> None:
    scores = compute_lm_score(["GROWTH"])
    expected = {
        "positive_ratio", "negative_ratio", "net_sentiment",
        "uncertainty_ratio", "litigious_ratio", "constraining_ratio", "n_tokens",
    }
    assert expected.issubset(set(scores.keys()))


@_patch_lm
def test_compute_score_n_tokens_is_correct(*_) -> None:
    tokens = ["GROWTH", "RISK", "COMPANY"]
    scores = compute_lm_score(tokens)
    assert scores["n_tokens"] == 3


@_patch_lm
def test_compute_score_counts_duplicates(*_) -> None:
    # RISK appears twice → negative_ratio should be 2/4 = 0.5
    tokens = ["RISK", "RISK", "COMPANY", "QUARTER"]
    scores = compute_lm_score(tokens)
    assert abs(scores["negative_ratio"] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# classify_sentiment
# ---------------------------------------------------------------------------

def test_classify_positive() -> None:
    assert classify_sentiment({"net_sentiment": 0.05}) == "positive"


def test_classify_negative() -> None:
    assert classify_sentiment({"net_sentiment": -0.05}) == "negative"


def test_classify_neutral_above_zero() -> None:
    assert classify_sentiment({"net_sentiment": 0.005}, threshold=0.01) == "neutral"


def test_classify_neutral_at_zero() -> None:
    assert classify_sentiment({"net_sentiment": 0.0}) == "neutral"


def test_classify_custom_threshold() -> None:
    assert classify_sentiment({"net_sentiment": 0.03}, threshold=0.05) == "neutral"
    assert classify_sentiment({"net_sentiment": 0.06}, threshold=0.05) == "positive"


# ---------------------------------------------------------------------------
# score_dataframe
# ---------------------------------------------------------------------------

@_patch_lm
def test_score_dataframe_adds_label_column(*_) -> None:
    df = pd.DataFrame({
        "ticker": ["AAPL"],
        "tokens": [["GROWTH", "PROFIT", "STRONG"]],
    })
    result = score_dataframe(df, token_col="tokens", prefix="pr_lm")
    assert "pr_lm_label" in result.columns


@_patch_lm
def test_score_dataframe_does_not_mutate_input(*_) -> None:
    df = pd.DataFrame({"tokens": [["GROWTH"]]})
    original_cols = list(df.columns)
    score_dataframe(df, token_col="tokens", prefix="lm")
    assert list(df.columns) == original_cols


@_patch_lm
def test_score_dataframe_correct_prefix(*_) -> None:
    df = pd.DataFrame({"tokens": [["RISK"]]})
    result = score_dataframe(df, token_col="tokens", prefix="qa_lm")
    assert "qa_lm_net_sentiment" in result.columns
    assert "qa_lm_label" in result.columns
