"""Regex-based segmentation of earnings call transcripts into
Prepared Remarks and Q&A sections."""

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Tier 1 — primary headers that Motley Fool uses consistently
_PREPARED_PATTERN = re.compile(
    r"prepared\s+remarks",
    re.IGNORECASE,
)

_QA_PATTERN = re.compile(
    r"(?:"
    r"questions?\s+(?:and\s+)?answers?"
    r"|q\s*(?:and|&)\s*a\s*(?:session)?"
    r"|question[- ]and[- ]answer"
    r"|open\s+(?:the\s+)?(?:floor|lines?)\s+for\s+questions?"
    r"|we\s+(?:will\s+)?(?:now\s+)?(?:begin|open|take)\s+(?:the\s+)?q(?:uestion)?s?"
    r")",
    re.IGNORECASE,
)

# Tier 2 fallback — Operator speaker turns signal the Q&A boundary.
# Motley Fool uses two formats: old = "Operator:" (with colon),
# new = "Operator\n" (speaker name on its own line, no colon).
_OPERATOR_TURN = re.compile(
    r"(?:^|\n)Operator\s*(?:\n|:)",
    re.IGNORECASE,
)


@dataclass
class TranscriptSegments:
    """Container for a split earnings call transcript.

    Attributes:
        ticker: Stock ticker symbol.
        date: Earnings call date as YYYY-MM-DD string.
        prepared_remarks: Text of the prepared remarks section.
        qa_section: Text of the Q&A section.
        split_successful: True when a reliable split was achieved.
        split_method: One of 'regex_primary', 'regex_fallback', 'heuristic'.
    """

    ticker: str
    date: str
    prepared_remarks: str
    qa_section: str
    split_successful: bool
    split_method: str


def split_transcript(
    raw_text: str,
    ticker: str,
    date: str,
) -> TranscriptSegments:
    """Split a transcript into Prepared Remarks and Q&A sections.

    Three-tier strategy:
      1. Primary regex: look for explicit "Prepared Remarks" and
         "Questions and Answers" headers.
      2. Fallback regex: use the first "Operator:" speaker turn after
         the first third of the text as the Q&A boundary.
      3. Heuristic: split at 60% of the text length.
         split_successful is set to False for heuristic splits.
         These rows are EXCLUDED from evaluation metrics but are
         LOGGED as failure cases in the error analysis report, satisfying
         the mandatory 10-sample failure analysis requirement.

    Args:
        raw_text: Full transcript text.
        ticker: Stock ticker (stored in the result for traceability).
        date: Earnings call date YYYY-MM-DD.

    Returns:
        TranscriptSegments dataclass.
    """
    # --- Tier 1: primary regex -------------------------------------------------
    n = len(raw_text)
    pr_match = _PREPARED_PATTERN.search(raw_text)
    qa_match = _QA_PATTERN.search(raw_text)

    # Require QA marker before 85% to avoid matching closing lines like
    # "This concludes our question-and-answer session" at the very end.
    if (pr_match and qa_match
            and pr_match.start() < qa_match.start()
            and qa_match.start() < n * 0.85):
        prepared = raw_text[pr_match.end(): qa_match.start()].strip()
        qa = raw_text[qa_match.end():].strip()
        return TranscriptSegments(
            ticker=ticker,
            date=date,
            prepared_remarks=prepared,
            qa_section=qa,
            split_successful=True,
            split_method="regex_primary",
        )

    # --- Tier 2: operator-turn fallback ----------------------------------------
    # Find the first "Operator:" turn that appears after the first third of the text.
    midpoint = len(raw_text) // 3
    for m in _OPERATOR_TURN.finditer(raw_text):
        if m.start() > midpoint:
            prepared = raw_text[:m.start()].strip()
            qa = raw_text[m.start():].strip()
            return TranscriptSegments(
                ticker=ticker,
                date=date,
                prepared_remarks=prepared,
                qa_section=qa,
                split_successful=True,
                split_method="regex_fallback",
            )

    # --- Tier 3: heuristic split -----------------------------------------------
    # split_successful=False — excluded from evaluation, logged as failure case.
    split_idx = int(len(raw_text) * 0.60)
    return TranscriptSegments(
        ticker=ticker,
        date=date,
        prepared_remarks=raw_text[:split_idx].strip(),
        qa_section=raw_text[split_idx:].strip(),
        split_successful=False,
        split_method="heuristic",
    )


def validate_split(segments: TranscriptSegments, min_chars: int = 500) -> bool:
    """Return True if both sections meet a minimum character threshold.

    Args:
        segments: Result of split_transcript().
        min_chars: Minimum number of characters required in each section.

    Returns:
        True if both sections are long enough to be meaningful.
    """
    return (
        len(segments.prepared_remarks) >= min_chars
        and len(segments.qa_section) >= min_chars
    )
