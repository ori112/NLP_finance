"""Tests for FinBERT chunker and inference pipeline.

Fast tests (no model downloads): chunker logic only.
Slow tests (require model download): marked @pytest.mark.slow
Run fast tests only: pytest -m "not slow"
"""

import pytest

from src.models.chunker import aggregate_chunk_scores, sliding_window_chunks


# ---------------------------------------------------------------------------
# sliding_window_chunks — fast, no model needed
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal tokenizer stub for chunker tests — no HuggingFace download needed."""

    def __call__(self, text, add_special_tokens=False, truncation=False,
                 return_attention_mask=False, return_token_type_ids=False):
        # Tokenize by splitting on whitespace — simple but enough for unit tests
        tokens = text.split()
        return {"input_ids": list(range(len(tokens)))}

    def decode(self, token_ids, skip_special_tokens=True):
        # Return a placeholder string proportional to chunk length
        return " ".join(f"word{i}" for i in token_ids)


_TOK = _FakeTokenizer()


def test_short_text_produces_single_chunk() -> None:
    text = "short text"  # 2 tokens — well under any limit
    chunks = sliding_window_chunks(text, _TOK, max_tokens=512, stride=128)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks() -> None:
    # 600 "words" — exceeds the 510-token effective limit
    text = " ".join(f"word{i}" for i in range(600))
    chunks = sliding_window_chunks(text, _TOK, max_tokens=512, stride=128)
    assert len(chunks) > 1


def test_chunk_count_is_deterministic() -> None:
    text = " ".join(f"word{i}" for i in range(600))
    chunks1 = sliding_window_chunks(text, _TOK, max_tokens=512, stride=128)
    chunks2 = sliding_window_chunks(text, _TOK, max_tokens=512, stride=128)
    assert len(chunks1) == len(chunks2)


def test_smaller_stride_produces_more_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(600))
    chunks_64 = sliding_window_chunks(text, _TOK, max_tokens=512, stride=64)
    chunks_256 = sliding_window_chunks(text, _TOK, max_tokens=512, stride=256)
    assert len(chunks_64) > len(chunks_256)


def test_empty_text_produces_one_chunk() -> None:
    chunks = sliding_window_chunks("", _TOK)
    assert len(chunks) == 1


# ---------------------------------------------------------------------------
# aggregate_chunk_scores
# ---------------------------------------------------------------------------

def test_aggregate_mean_averages_correctly() -> None:
    scores = [
        {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
        {"positive": 0.6, "negative": 0.3, "neutral": 0.1},
    ]
    result = aggregate_chunk_scores(scores, strategy="mean")
    assert abs(result["positive"] - 0.7) < 1e-6
    # Binary task: positive_prob > negative_prob → "up"
    assert result["label"] == "up"


def test_aggregate_mean_single_chunk_down() -> None:
    scores = [{"positive": 0.1, "negative": 0.8, "neutral": 0.1}]
    result = aggregate_chunk_scores(scores, strategy="mean")
    assert result["label"] == "down"


def test_aggregate_majority_uses_per_chunk_direction() -> None:
    # Two chunks vote up (positive>negative), one votes down — majority "up".
    scores = [
        {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
        {"positive": 0.6, "negative": 0.2, "neutral": 0.2},
        {"positive": 0.1, "negative": 0.7, "neutral": 0.2},
    ]
    result = aggregate_chunk_scores(scores, strategy="majority")
    assert result["label"] == "up"


def test_aggregate_empty_input_returns_down() -> None:
    # Empty input is a degenerate case — convention: default to "down".
    result = aggregate_chunk_scores([])
    assert result["label"] == "down"


def test_aggregate_invalid_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unknown aggregation strategy"):
        aggregate_chunk_scores([{"positive": 1.0, "negative": 0.0, "neutral": 0.0}], strategy="bad")


# ---------------------------------------------------------------------------
# FinBERT integration test — requires model download
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_predict_sentiment_returns_valid_label() -> None:
    from src.models.finbert import load_finbert_pipeline, predict_sentiment

    pipe = load_finbert_pipeline(device=-1)
    result = predict_sentiment("Revenue grew strongly this quarter.", pipe)
    assert result["label"] in {"up", "down"}
    total = result["positive"] + result["negative"] + result["neutral"]
    assert abs(total - 1.0) < 0.05  # probabilities sum to ~1


@pytest.mark.slow
def test_predict_sentiment_n_chunks_for_short_text() -> None:
    from src.models.finbert import load_finbert_pipeline, predict_sentiment

    pipe = load_finbert_pipeline(device=-1)
    result = predict_sentiment("Short sentence.", pipe)
    assert result["n_chunks"] == 1
