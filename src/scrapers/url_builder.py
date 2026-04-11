"""URL construction and manifest loading for Motley Fool transcript scraping."""

import csv
from pathlib import Path

MANIFEST_PATH = Path("data/raw/url_manifest.csv")


def build_transcript_url(slug: str, year: int, month: int, day: int) -> str:
    """Construct a canonical Motley Fool transcript URL from its components.

    Args:
        slug: URL slug for the transcript (e.g. 'apple-q1-2024-earnings-call-transcript').
        year: Four-digit year of the earnings call.
        month: Month (1-12) of the earnings call.
        day: Day (1-31) of the earnings call.

    Returns:
        Full URL string.
    """
    return (
        f"https://www.fool.com/earnings/call-transcripts/"
        f"{year}/{month:02d}/{day:02d}/{slug}/"
    )


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    """Load the pre-curated URL manifest CSV.

    Expected CSV columns: ticker, company, date (YYYY-MM-DD), url, sector

    Args:
        manifest_path: Path to the manifest CSV file.

    Returns:
        List of dicts, one per transcript entry.

    Raises:
        FileNotFoundError: If the manifest CSV does not exist.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"URL manifest not found at {manifest_path}. "
            "Create data/raw/url_manifest.csv before running the scraper."
        )

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
