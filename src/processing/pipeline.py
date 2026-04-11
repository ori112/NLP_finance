"""Preprocessing pipeline: segments transcripts, cleans text, fetches returns,
and produces the master segments DataFrame used by all model phases."""

import logging
from pathlib import Path

import pandas as pd

from src.processing.cleaner import clean_text
from src.processing.returns import build_returns_dataframe
from src.processing.segmenter import split_transcript, validate_split
from src.processing.tokenizer import tokenize_for_lexicon
from src.utils.storage import list_transcripts, load_transcript

logger = logging.getLogger(__name__)

_SEGMENTS_OUTPUT = Path("data/processed/segments.parquet")


def run_preprocessing(
    raw_dir: Path = Path("data/raw/transcripts"),
    output_path: Path = _SEGMENTS_OUTPUT,
    min_chars: int = 500,
) -> pd.DataFrame:
    """Full preprocessing pipeline for all scraped transcripts.

    Steps:
      1. Load all transcript JSON files from raw_dir.
      2. Split each into Prepared Remarks and Q&A sections.
      3. Clean and tokenize both segments.
      4. Fetch market-adjusted returns via yfinance (cached).
      5. Merge everything into a single DataFrame.
      6. Save to output_path as Parquet.

    Column schema of output DataFrame:
      ticker, date, company,
      prepared_text, qa_text,          # cleaned text for FinBERT
      prepared_tokens, qa_tokens,      # uppercase token lists for LM scoring
      return_1d, return_3d,
      market_adj_1d, market_adj_3d,
      label_1d, label_3d,
      split_successful, split_method

    Args:
        raw_dir: Directory containing transcript JSON files.
        output_path: Where to save the segments Parquet file.
        min_chars: Minimum characters for a section to be considered valid.

    Returns:
        Master segments DataFrame.
    """
    paths = list_transcripts(raw_dir)
    if not paths:
        raise FileNotFoundError(
            f"No transcript JSON files found in {raw_dir}. "
            "Run `python main.py --mode scrape` first."
        )

    logger.info("Processing %d transcripts …", len(paths))
    records = [load_transcript(p) for p in paths]

    rows = []
    split_stats = {"regex_primary": 0, "regex_fallback": 0, "heuristic": 0}

    for rec in records:
        segments = split_transcript(rec["raw_text"], rec["ticker"], rec["date"])
        split_stats[segments.split_method] = split_stats.get(segments.split_method, 0) + 1

        if not validate_split(segments, min_chars):
            logger.warning(
                "Skipping %s %s — segment too short after split (%s).",
                rec["ticker"],
                rec["date"],
                segments.split_method,
            )
            continue

        rows.append(
            {
                "ticker": rec["ticker"],
                "date": rec["date"],
                "company": rec.get("company", ""),
                "prepared_text": clean_text(segments.prepared_remarks),
                "qa_text": clean_text(segments.qa_section),
                "prepared_tokens": tokenize_for_lexicon(segments.prepared_remarks),
                "qa_tokens": tokenize_for_lexicon(segments.qa_section),
                "split_successful": segments.split_successful,
                "split_method": segments.split_method,
            }
        )

    logger.info(
        "Split statistics — primary: %d | fallback: %d | heuristic: %d",
        split_stats.get("regex_primary", 0),
        split_stats.get("regex_fallback", 0),
        split_stats.get("heuristic", 0),
    )

    df = pd.DataFrame(rows)

    # Attach market-adjusted returns
    returns_df = build_returns_dataframe(records)
    df = df.merge(returns_df, on=["ticker", "date"], how="left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info("Segments saved to %s (%d rows).", output_path, len(df))

    return df
