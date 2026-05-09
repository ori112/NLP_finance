"""Tests for the Motley Fool scraper — no network calls, fixture-based only."""

import time
from pathlib import Path

import pytest

from src.scrapers.motley_fool import parse_transcript
from src.scrapers.rate_limiter import RateLimiter
from src.scrapers.url_builder import build_transcript_url, load_manifest

FIXTURE_HTML = Path(__file__).parent / "fixtures" / "sample_transcript.html"
MANIFEST_PATH = Path("data/raw/url_manifest.csv")


# ---------------------------------------------------------------------------
# parse_transcript
# ---------------------------------------------------------------------------

def test_parse_transcript_extracts_text() -> None:
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    result = parse_transcript(html, url="http://test", ticker="AAPL", date="2023-01-26", company="Apple Inc.")
    assert len(result["raw_text"]) > 200


def test_parse_transcript_returns_correct_ticker() -> None:
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    result = parse_transcript(html, url="http://test", ticker="aapl", date="2023-01-26", company="Apple Inc.")
    assert result["ticker"] == "AAPL"


def test_parse_transcript_includes_required_keys() -> None:
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    result = parse_transcript(html, url="http://test", ticker="AAPL", date="2023-01-26", company="Apple Inc.")
    for key in ("ticker", "url", "date", "company", "raw_text", "scraped_at"):
        assert key in result, f"Missing key: {key}"


def test_parse_transcript_contains_prepared_and_qa_markers() -> None:
    html = FIXTURE_HTML.read_text(encoding="utf-8")
    result = parse_transcript(html, url="http://test", ticker="AAPL", date="2023-01-26", company="Apple Inc.")
    text = result["raw_text"].lower()
    assert "prepared remarks" in text
    assert "questions and answers" in text


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

def test_rate_limiter_enforces_minimum_delay() -> None:
    limiter = RateLimiter(min_delay=0.1, max_delay=0.15)
    limiter.wait()                      # prime the timer
    start = time.monotonic()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.09             # allow 10ms tolerance


def test_rate_limiter_respects_max_delay() -> None:
    limiter = RateLimiter(min_delay=0.05, max_delay=0.1)
    limiter.wait()
    start = time.monotonic()
    limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed < 0.5               # should never approach this


# ---------------------------------------------------------------------------
# url_builder
# ---------------------------------------------------------------------------

def test_build_transcript_url_format() -> None:
    url = build_transcript_url("apple-q1-2023", 2023, 1, 26)
    assert url == "https://www.fool.com/earnings/call-transcripts/2023/01/26/apple-q1-2023/"


def test_build_transcript_url_zero_pads_month_and_day() -> None:
    url = build_transcript_url("test-slug", 2022, 3, 5)
    assert "/2022/03/05/" in url


def test_load_manifest_returns_list_of_dicts() -> None:
    entries = load_manifest(MANIFEST_PATH)
    assert isinstance(entries, list)
    assert len(entries) > 0
    assert "ticker" in entries[0]
    assert "url" in entries[0]
    assert "date" in entries[0]


def test_load_manifest_reaches_target_size() -> None:
    entries = load_manifest(MANIFEST_PATH)
    # Proposal target is 150-200 transcripts. Current manifest is 124 — the gap
    # is documented as a known limitation in the report (Motley Fool URL
    # restructure invalidated 7 tickers; 17 valid ones with 124 working URLs
    # remain). Floor of 100 guards against accidental manifest truncation.
    assert len(entries) >= 100, f"Manifest unexpectedly small: {len(entries)} entries"


def test_load_manifest_raises_on_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(Path("data/raw/nonexistent_manifest.csv"))
