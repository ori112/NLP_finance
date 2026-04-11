"""Sliding window chunking for long texts that exceed BERT's 512-token limit."""

from transformers import PreTrainedTokenizerBase


def sliding_window_chunks(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int = 512,
    stride: int = 128,
    special_tokens: int = 2,
) -> list[str]:
    """Split text into overlapping token-bounded chunks.

    Why overlap matters: sentiment-bearing phrases near a chunk boundary
    would be truncated in a hard-split approach. With stride=128, each
    boundary region appears in two consecutive chunks, so FinBERT scores
    it at least once in full context. The chunk scores are later averaged.

    With max_tokens=512 and stride=128:
      - Effective chunk size: 510 tokens (512 - 2 special tokens)
      - Overlap between consecutive chunks: 510 - 128 = 382 tokens (75%)
      - Texts shorter than 510 tokens produce a single chunk (no splitting)

    Args:
        text: Clean segment text. Do NOT remove stop words — BERT needs them.
        tokenizer: HuggingFace PreTrainedTokenizer (e.g. FinBERT tokenizer).
        max_tokens: Maximum tokens per chunk INCLUDING [CLS] and [SEP].
        stride: Number of tokens to advance the window start each step.
        special_tokens: Number of special tokens ([CLS] + [SEP] = 2).

    Returns:
        List of decoded text strings, one per chunk. Always at least 1 element.
    """
    effective_max = max_tokens - special_tokens

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids: list[int] = encoding["input_ids"]

    if len(token_ids) <= effective_max:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + effective_max, len(token_ids))
        chunk_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(token_ids):
            break
        start += stride

    return chunks


def aggregate_chunk_scores(
    chunk_scores: list[dict[str, float]],
    strategy: str = "mean",
) -> dict[str, float]:
    """Combine per-chunk FinBERT scores into a single document-level score.

    Args:
        chunk_scores: List of dicts, each with keys 'positive', 'negative', 'neutral'
                      and float probability values (should sum to ~1.0 per chunk).
        strategy: Aggregation strategy — one of:
          'mean'     : Simple average of probabilities across all chunks (default).
          'weighted' : Weight each chunk by its token count (not yet implemented;
                       falls back to 'mean' — extend when chunk lengths are tracked).
          'majority' : Pick the label that appears most often across chunks.

    Returns:
        Dict with keys 'positive', 'negative', 'neutral' (probabilities) and 'label'.
    """
    if not chunk_scores:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "label": "neutral"}

    labels = ["positive", "negative", "neutral"]

    if strategy in ("mean", "weighted"):
        avg = {
            lbl: sum(c.get(lbl, 0.0) for c in chunk_scores) / len(chunk_scores)
            for lbl in labels
        }
        avg["label"] = max(labels, key=lambda l: avg[l])
        return avg

    if strategy == "majority":
        label_counts: dict[str, int] = {lbl: 0 for lbl in labels}
        for c in chunk_scores:
            winner = max(labels, key=lambda l: c.get(l, 0.0))
            label_counts[winner] += 1
        majority_label = max(label_counts, key=lambda l: label_counts[l])
        # Return mean probabilities but majority label
        avg = {
            lbl: sum(c.get(lbl, 0.0) for c in chunk_scores) / len(chunk_scores)
            for lbl in labels
        }
        avg["label"] = majority_label
        return avg

    raise ValueError(f"Unknown aggregation strategy: {strategy!r}. Use 'mean', 'weighted', or 'majority'.")
