"""Motley Fool earnings call transcript scraper."""

import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.scrapers.rate_limiter import RateLimiter
from src.scrapers.url_builder import MANIFEST_PATH, load_manifest
from src.utils.storage import RAW_DIR, list_transcripts, save_transcript

logger = logging.getLogger(__name__)

_ua = UserAgent()
_limiter = RateLimiter(min_delay=3.0, max_delay=8.0)


def _make_session() -> requests.Session:
    """Create a requests Session with a randomised User-Agent header."""
    session = requests.Session()
    session.headers.update({"User-Agent": _ua.random})
    return session


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    reraise=True,
)
def fetch_page(url: str, session: requests.Session) -> str:
    """Fetch raw HTML for a transcript page with automatic retry and backoff.

    Args:
        url: Full URL to fetch.
        session: Requests session (carries headers + cookies).

    Returns:
        Raw HTML string.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status after retries.
    """
    _limiter.wait()
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_transcript(html: str, url: str, ticker: str, date: str, company: str) -> dict:
    """Parse a Motley Fool transcript page into a structured record.

    Motley Fool wraps the transcript body inside <div class="article-body">
    (or similar). We extract all paragraph text and join with newlines.

    Args:
        html: Raw HTML of the transcript page.
        url: Source URL (stored for provenance).
        ticker: Stock ticker symbol.
        date: Earnings call date as YYYY-MM-DD string.
        company: Company name.

    Returns:
        Dict with keys: ticker, url, date, company, raw_text, scraped_at.
    """
    soup = BeautifulSoup(html, "lxml")

    # Motley Fool article body selector — primary then fallback
    body = soup.find("div", class_="article-body")
    if body is None:
        body = soup.find("div", {"id": "fool-article-body"})
    if body is None:
        # Last-resort: grab all <p> tags in the page
        body = soup

    paragraphs = body.find_all("p")
    raw_text = "\n".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

    return {
        "ticker": ticker.upper(),
        "url": url,
        "date": date,
        "company": company,
        "raw_text": raw_text,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def _already_scraped(ticker: str, date: str) -> bool:
    """Return True if a transcript JSON for this ticker+date already exists."""
    path = RAW_DIR / f"{ticker.upper()}_{date}.json"
    return path.exists()


def run_scraper(tickers: list[str] | None = None) -> None:
    """Entry point: load the URL manifest and scrape missing transcripts.

    Skips any ticker+date pair that already has a JSON file on disk so the
    scraper is safe to re-run after interruptions.

    Args:
        tickers: Optional list of ticker symbols to restrict scraping to.
                 If None, all entries in the manifest are processed.
    """
    entries = load_manifest(MANIFEST_PATH)

    if tickers:
        tickers_upper = {t.upper() for t in tickers}
        entries = [e for e in entries if e["ticker"].upper() in tickers_upper]

    total = len(entries)
    logger.info("Manifest loaded: %d entries to process.", total)
    session = _make_session()
    skipped = 0
    saved = 0
    failed = 0

    for i, entry in enumerate(tqdm(entries, desc="Scraping transcripts")):
        ticker = entry["ticker"].strip().upper()
        date = entry["date"].strip()
        url = entry["url"].strip()
        company = entry.get("company", "").strip()

        if _already_scraped(ticker, date):
            skipped += 1
            logger.info("[%d/%d] SKIP  %s %s (already on disk)", i + 1, total, ticker, date)
            continue

        try:
            logger.info("[%d/%d] FETCH %s %s …", i + 1, total, ticker, date)
            html = fetch_page(url, session)
            record = parse_transcript(html, url, ticker, date, company)

            if len(record["raw_text"]) < 200:
                logger.warning("[%d/%d] SHORT %s %s — skipping (text < 200 chars).", i + 1, total, ticker, date)
                failed += 1
                continue

            save_transcript(record)
            saved += 1
            logger.info("[%d/%d] SAVED %s %s (%d chars)", i + 1, total, ticker, date, len(record["raw_text"]))

        except Exception as exc:
            logger.error("[%d/%d] FAIL  %s %s — %s", i + 1, total, ticker, date, exc)
            failed += 1

    logger.info(
        "Scraping complete. Saved: %d | Skipped (cached): %d | Failed: %d",
        saved,
        skipped,
        failed,
    )
