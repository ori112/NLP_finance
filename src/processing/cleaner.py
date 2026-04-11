"""Text cleaning utilities for earnings call transcripts.

Note on stop words: stop word removal is intentionally NOT done here.
- clean_text() output is consumed by FinBERT, which needs natural sentence structure.
- Stop words are removed in tokenizer.py, which is used only for LM Lexicon scoring.
"""

import re


# Matches "Firstname Lastname -- Title:" style speaker labels
_SPEAKER_LABEL = re.compile(
    r"^[A-Z][a-zA-Z\s\-']+(?:--|—)[^\n:]+:\s*",
    re.MULTILINE,
)

# Matches "Operator:" alone on a line
_OPERATOR_LABEL = re.compile(r"^\s*Operator\s*:\s*", re.MULTILINE | re.IGNORECASE)

# Collapse 3+ newlines into 2
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Collapse runs of spaces/tabs (not newlines) into single space
_EXCESS_SPACES = re.compile(r"[ \t]{2,}")

# Common Motley Fool boilerplate patterns
_BOILERPLATE = re.compile(
    r"(?:all rights reserved|fool\.com|the motley fool|"
    r"this (?:transcript|article) (?:is|was) prepared|"
    r"image source|advertisement)",
    re.IGNORECASE,
)


def remove_speaker_labels(text: str) -> str:
    """Strip 'Firstname Lastname -- Title:' and 'Operator:' prefixes.

    Args:
        text: Raw transcript text with speaker labels on each paragraph.

    Returns:
        Text with speaker attribution lines removed.
    """
    text = _SPEAKER_LABEL.sub("", text)
    text = _OPERATOR_LABEL.sub("", text)
    return text


def remove_boilerplate(text: str) -> str:
    """Remove lines containing known Motley Fool boilerplate phrases.

    Args:
        text: Transcript text.

    Returns:
        Text with boilerplate lines dropped.
    """
    lines = text.splitlines()
    cleaned = [ln for ln in lines if not _BOILERPLATE.search(ln)]
    return "\n".join(cleaned)


def normalize_whitespace(text: str) -> str:
    """Collapse excess whitespace while preserving paragraph breaks.

    Args:
        text: Text with potentially irregular spacing.

    Returns:
        Normalized text with single spaces and at most double newlines.
    """
    text = _EXCESS_SPACES.sub(" ", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline: remove labels, boilerplate, normalize whitespace.

    Stop words are NOT removed here — FinBERT requires natural sentence structure.
    For LM Lexicon scoring, use tokenizer.tokenize_for_lexicon() which handles
    stop word removal separately.

    Args:
        text: Raw segment text.

    Returns:
        Clean text ready for tokenization or transformer input.
    """
    text = remove_speaker_labels(text)
    text = remove_boilerplate(text)
    text = normalize_whitespace(text)
    return text
