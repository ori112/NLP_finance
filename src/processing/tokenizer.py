"""Tokenization utilities for LM Lexicon scoring and FinBERT chunking."""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data on first import (silent if already present)
for _resource in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{_resource}" if "punkt" in _resource else f"corpora/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)

_ENGLISH_STOPWORDS: set[str] = set(stopwords.words("english"))
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "''""–—")


def tokenize_for_lexicon(text: str, remove_stops: bool = True) -> list[str]:
    """Tokenize and normalise text for Loughran-McDonald scoring.

    Pipeline:
      1. Lowercase
      2. Strip punctuation
      3. Word-tokenize (NLTK punkt)
      4. Optionally remove English stop words
      5. Convert to UPPERCASE (LM dictionary keys are all uppercase)
      6. Drop tokens shorter than 2 characters

    Note: Financial stop words differ from general English — common words
    like "risk", "loss", "gain" carry meaning in the LM lexicon so we use
    the standard NLTK English stop word list, which does NOT include those.

    Args:
        text: Cleaned segment text.
        remove_stops: If True, remove standard English stop words before
                      returning tokens (recommended for LM scoring).

    Returns:
        List of uppercase tokens ready for LM dictionary lookup.
    """
    # Lowercase and strip punctuation
    text = text.lower().translate(_PUNCT_TABLE)

    tokens = word_tokenize(text)

    if remove_stops:
        tokens = [t for t in tokens if t not in _ENGLISH_STOPWORDS]

    # Uppercase for LM dict lookup; drop very short tokens
    return [t.upper() for t in tokens if len(t) >= 2 and t.isalpha()]


def chunk_for_bert(
    text: str,
    tokenizer,
    max_tokens: int = 512,
    stride: int = 128,
) -> list[str]:
    """Split text into overlapping chunks fitting within max_tokens.

    Uses a sliding window so that sentiment signals near chunk boundaries
    appear in at least two consecutive chunks. Each chunk is decoded back
    to a string for the HuggingFace pipeline API.

    Args:
        text: Clean segment text (do NOT remove stop words — BERT needs them).
        tokenizer: A HuggingFace PreTrainedTokenizer instance.
        max_tokens: Maximum tokens per chunk including special tokens
                    [CLS] and [SEP] (default 512 = BERT limit).
        stride: Number of tokens to advance between chunk start positions.
                Lower stride = more overlap = more compute but finer coverage.

    Returns:
        List of decoded text strings, one per chunk.
    """
    effective_max = max_tokens - 2  # reserve 2 slots for [CLS] and [SEP]

    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        truncation=False,
    )
    token_ids: list[int] = encoding["input_ids"]

    if len(token_ids) <= effective_max:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        end = min(start + effective_max, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(token_ids):
            break
        start += stride

    return chunks
