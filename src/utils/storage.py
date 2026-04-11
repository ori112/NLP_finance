"""Helpers for saving and loading transcript JSON files."""

import json
from pathlib import Path

RAW_DIR = Path("data/raw/transcripts")


def save_transcript(record: dict) -> Path:
    """Save a transcript record as a JSON file.

    Filename format: {TICKER}_{YYYY-MM-DD}.json

    Args:
        record: Dict with at minimum keys 'ticker' and 'date'.

    Returns:
        Path to the saved file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ticker = record["ticker"].upper()
    date = record["date"]
    path = RAW_DIR / f"{ticker}_{date}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def load_transcript(path: Path) -> dict:
    """Load a single transcript JSON file.

    Args:
        path: Path to a transcript JSON file.

    Returns:
        Parsed transcript dict.
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_transcripts(directory: Path = RAW_DIR) -> list[Path]:
    """Return all transcript JSON paths in a directory, sorted by filename.

    Args:
        directory: Directory to scan for JSON files.

    Returns:
        Sorted list of Path objects.
    """
    if not directory.exists():
        return []
    return sorted(directory.glob("*.json"))
