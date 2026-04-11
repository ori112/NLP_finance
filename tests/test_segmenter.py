"""Tests for transcript segmentation and text cleaning."""

from src.processing.cleaner import clean_text, normalize_whitespace, remove_speaker_labels
from src.processing.segmenter import TranscriptSegments, split_transcript, validate_split
from src.processing.tokenizer import tokenize_for_lexicon

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FULL_TRANSCRIPT = """
Welcome to the Q4 2023 earnings call.

Prepared Remarks

Tim Cook -- Chief Executive Officer: We had a strong quarter with revenue of $117 billion.
Services reached record highs and our installed base surpassed 2 billion devices.
We are very pleased with the results and remain focused on innovation and growth opportunities.

Luca Maestri -- Chief Financial Officer: Gross margin was 42.9 percent.
We returned 25 billion dollars to shareholders through buybacks and dividends.
Our guidance for next quarter reflects continued strong demand.

Questions and Answers

Operator: Our first question comes from Katie Huberty at Morgan Stanley.
Katie Huberty -- Morgan Stanley -- Analyst: Can you talk about iPhone demand trends?
Tim Cook -- Chief Executive Officer: Demand has been strong across all models.
"""

_NO_MARKERS_TRANSCRIPT = "This text has absolutely no section markers at all. " * 50

_OPERATOR_ONLY_TRANSCRIPT = """
Welcome to the earnings call. Management will now present results.

Revenue grew significantly this quarter driven by strong performance in all segments.
We are investing heavily in research and development to sustain long-term growth.
Our cash position remains strong and we continue to return capital to shareholders.

Operator: We will now take questions from analysts.
Analyst: What is your outlook for next quarter?
Management: We expect continued growth across all business segments.
"""


# ---------------------------------------------------------------------------
# split_transcript — primary regex
# ---------------------------------------------------------------------------

def test_split_primary_identifies_both_sections() -> None:
    result = split_transcript(_FULL_TRANSCRIPT, "AAPL", "2023-11-02")
    assert result.split_successful is True
    assert result.split_method == "regex_primary"
    assert len(result.prepared_remarks) > 100
    assert len(result.qa_section) > 100


def test_split_primary_prepared_does_not_contain_qa_header() -> None:
    result = split_transcript(_FULL_TRANSCRIPT, "AAPL", "2023-11-02")
    assert "questions and answers" not in result.prepared_remarks.lower()


def test_split_primary_qa_contains_analyst_content() -> None:
    result = split_transcript(_FULL_TRANSCRIPT, "AAPL", "2023-11-02")
    assert "Katie Huberty" in result.qa_section or "question" in result.qa_section.lower()


# ---------------------------------------------------------------------------
# split_transcript — operator fallback
# ---------------------------------------------------------------------------

def test_split_fallback_uses_operator_boundary() -> None:
    result = split_transcript(_OPERATOR_ONLY_TRANSCRIPT, "TEST", "2023-01-01")
    assert result.split_successful is True
    assert result.split_method == "regex_fallback"
    assert "Operator" in result.qa_section


# ---------------------------------------------------------------------------
# split_transcript — heuristic fallback
# ---------------------------------------------------------------------------

def test_split_heuristic_flags_as_unsuccessful() -> None:
    result = split_transcript(_NO_MARKERS_TRANSCRIPT, "TEST", "2023-01-01")
    assert result.split_successful is False
    assert result.split_method == "heuristic"


def test_split_heuristic_still_produces_non_empty_sections() -> None:
    result = split_transcript(_NO_MARKERS_TRANSCRIPT, "TEST", "2023-01-01")
    assert len(result.prepared_remarks) > 0
    assert len(result.qa_section) > 0


def test_split_stores_ticker_and_date() -> None:
    result = split_transcript(_FULL_TRANSCRIPT, "AAPL", "2023-11-02")
    assert result.ticker == "AAPL"
    assert result.date == "2023-11-02"


# ---------------------------------------------------------------------------
# validate_split
# ---------------------------------------------------------------------------

def test_validate_split_passes_for_long_sections() -> None:
    seg = TranscriptSegments(
        ticker="X", date="2023-01-01",
        prepared_remarks="a" * 600,
        qa_section="b" * 600,
        split_successful=True,
        split_method="regex_primary",
    )
    assert validate_split(seg, min_chars=500) is True


def test_validate_split_fails_for_short_section() -> None:
    seg = TranscriptSegments(
        ticker="X", date="2023-01-01",
        prepared_remarks="short",
        qa_section="b" * 600,
        split_successful=True,
        split_method="regex_primary",
    )
    assert validate_split(seg, min_chars=500) is False


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

def test_clean_text_removes_speaker_labels() -> None:
    text = "Tim Cook -- Chief Executive Officer: Revenue was strong.\nWe had a great quarter."
    cleaned = clean_text(text)
    assert "Tim Cook" not in cleaned
    assert "Revenue was strong" in cleaned


def test_normalize_whitespace_collapses_spaces() -> None:
    text = "hello   world\n\n\n\nextra newlines"
    result = normalize_whitespace(text)
    assert "   " not in result
    assert result.count("\n") <= 2


# ---------------------------------------------------------------------------
# tokenize_for_lexicon
# ---------------------------------------------------------------------------

def test_tokenize_returns_uppercase() -> None:
    tokens = tokenize_for_lexicon("Revenue grew strongly this quarter.")
    assert all(t == t.upper() for t in tokens)


def test_tokenize_removes_punctuation() -> None:
    tokens = tokenize_for_lexicon("Growth: 12%, profit!")
    assert all(c.isalpha() or c == "" for t in tokens for c in t)


def test_tokenize_removes_stopwords_by_default() -> None:
    tokens = tokenize_for_lexicon("the company and its revenue")
    assert "THE" not in tokens
    assert "AND" not in tokens


def test_tokenize_keeps_stopwords_when_disabled() -> None:
    tokens = tokenize_for_lexicon("the company and its revenue", remove_stops=False)
    assert "THE" in tokens or "AND" in tokens


def test_tokenize_drops_single_char_tokens() -> None:
    tokens = tokenize_for_lexicon("a b c revenue")
    assert "A" not in tokens
    assert "B" not in tokens


def test_tokenize_financial_terms_not_removed() -> None:
    # "risk" and "loss" are NOT in NLTK English stopwords — they must survive
    tokens = tokenize_for_lexicon("significant risk and potential loss")
    assert "RISK" in tokens
    assert "LOSS" in tokens
