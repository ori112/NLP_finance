# Study Guide — Earnings Call Sentiment Analysis
### Deep Technical Reference for Oral Defense

This document walks through every concept in the codebase from first principles.
Each section covers *why* a decision was made, *how* it works mechanically, and
*what code implements it*. Read it once end-to-end before the defense; use
individual sections as a refresher for specific questions.

---

## Table of Contents

1. [Why Earnings Calls?](#1-why-earnings-calls)
2. [Research Question Design](#2-research-question-design)
3. [Web Scraping Architecture](#3-web-scraping-architecture)
4. [Rate Limiting and Polite Crawling](#4-rate-limiting-and-polite-crawling)
5. [Retry Logic and Exponential Backoff](#5-retry-logic-and-exponential-backoff)
6. [Manifest-Driven Scraping](#6-manifest-driven-scraping)
7. [HTML Parsing with BeautifulSoup](#7-html-parsing-with-beautifulsoup)
8. [Regex-Based Transcript Segmentation](#8-regex-based-transcript-segmentation)
9. [Text Cleaning Pipeline](#9-text-cleaning-pipeline)
10. [Tokenization — LM vs. BERT](#10-tokenization--lm-vs-bert)
11. [Market-Adjusted Returns (Ground Truth)](#11-market-adjusted-returns-ground-truth)
12. [Loughran-McDonald Lexicon](#12-loughran-mcdonald-lexicon)
13. [FinBERT: Architecture and Why It Exists](#13-finbert-architecture-and-why-it-exists)
14. [Sliding Window Chunking](#14-sliding-window-chunking)
15. [Evaluation Metrics](#15-evaluation-metrics)
16. [Error Analysis](#16-error-analysis)
17. [Visualizations](#17-visualizations)
18. [End-to-End Pipeline Flow](#18-end-to-end-pipeline-flow)
19. [Key Design Decisions Q&A](#19-key-design-decisions-qa)

---

## 1. Why Earnings Calls?

Earnings calls are quarterly events where a public company's management presents
financial results and then takes analyst questions. They are split into two
structurally distinct parts:

- **Prepared Remarks (PR):** A scripted monologue written by IR teams and lawyers.
  Language is carefully controlled. Executives choose every word.
- **Q&A Section:** Unscripted responses to analyst questions. Language is more
  spontaneous, hedging is more visible, and evasiveness or confidence is harder
  to control.

**Why is the PR vs. Q&A split scientifically interesting?**

The core hypothesis is: *if markets are informationally efficient, both sections
should carry equal predictive power — but if analysts extract incremental signal
from the Q&A that prepared text fails to reveal, QA sentiment should correlate
more strongly with returns.*

**Why this date range (2022–2023)?**

Sentiment variation is only scientifically useful if there is variance in *both*
directions. 2022–2023 provides the widest recent swing:
- 2022: Fed rate hikes (fastest in 40 years), Russia-Ukraine energy shock,
  tech selloff, SVB banking crisis
- 2023: AI boom (NVDA), partial recovery in tech, energy stabilization

A 2024-only dataset would be dominated by AI-positive sentiment, compressing
negative signal. Class imbalance (all-positive) would destroy classifier F1.

---

## 2. Research Question Design

**Research Question:**
> Which combination of (segment × model) best predicts 1-day market-adjusted
> stock returns following an earnings call?

The 4-way design matrix:

| Condition | Segment | Model |
|---|---|---|
| PR_LM | Prepared Remarks | Loughran-McDonald Lexicon |
| QA_LM | Q&A Section | Loughran-McDonald Lexicon |
| PR_FINBERT | Prepared Remarks | FinBERT |
| QA_FINBERT | Q&A Section | FinBERT |

This design is a 2×2 factorial experiment. It answers:
1. Does segment matter? (PR vs. QA — compare rows)
2. Does model sophistication matter? (LM vs. FinBERT — compare columns)
3. Is there an interaction? (maybe FinBERT only gains on QA but not PR)

**Ground truth:** Not a human label — it is the market's own judgment.
A stock that outperformed SPY by >0.5% in the day after the earnings call is
labeled "positive." This is a form of weak supervision.

```python
# src/processing/returns.py
def label_from_return(market_adj_return: float, threshold: float = 0.005) -> str:
    if market_adj_return > threshold:
        return "positive"
    if market_adj_return < -threshold:
        return "negative"
    return "neutral"
```

The ±0.5% threshold is a deliberate choice: small enough to capture real signal,
large enough to exclude noise from bid-ask spread and after-hours effects.

---

## 3. Web Scraping Architecture

### The stack

```
url_manifest.csv  →  url_builder.py  →  motley_fool.py  →  storage.py
   (data)             (load)             (fetch+parse)       (save JSON)
                                            ↑
                                    rate_limiter.py
                                    (polite delay)
```

### requests.Session — why a session, not bare requests.get()?

A `Session` object persists HTTP connection pooling and headers across requests:

```python
# src/scrapers/motley_fool.py
def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": _ua.random})
    return session
```

Benefits over `requests.get()`:
1. **TCP keep-alive**: Reuses the same underlying socket — faster, fewer handshakes
2. **Header persistence**: The User-Agent is set once, not per call
3. **Cookie handling**: If the site sets a session cookie on first request, it
   carries automatically to subsequent requests

### fake-useragent

`_ua = UserAgent()` pulls a random realistic browser string on each session
creation. This prevents the server from fingerprinting the scraper via a
constant `python-requests/2.x.x` header.

```python
# Typical output of _ua.random:
# "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36..."
```

---

## 4. Rate Limiting and Polite Crawling

### Why rate limit?

Without delays, a scraper can issue hundreds of requests per second — this is
indistinguishable from a DoS attack from the server's perspective. We impose
random delays to:
- Avoid overwhelming the server
- Appear like a human user
- Comply with the spirit of `robots.txt`
- Reduce the chance of IP-level rate limiting by the CDN

### Implementation — `RateLimiter`

```python
# src/scrapers/rate_limiter.py
class RateLimiter:
    def __init__(self, min_delay: float = 3.0, max_delay: float = 7.0) -> None:
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last_request: float = 0.0  # monotonic clock timestamp

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        delay = random.uniform(self.min_delay, self.max_delay)
        remaining = delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()
```

**Key design choices:**
- `time.monotonic()` instead of `time.time()`: Monotonic clock never goes
  backwards (e.g., clock adjustments, NTP sync). `time.time()` can jump.
- Jitter via `random.uniform`: If all bots waited exactly 5 seconds, they would
  still form a synchronized burst. Random jitter spreads the load.
- "Credit for elapsed work": If `fetch_page()` itself took 3 seconds (slow
  server), the actual sleep is `delay - 3`. We don't double-count server
  response time as "free delay."

### How it integrates with the scraper

```python
# src/scrapers/motley_fool.py
@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))
def fetch_page(url: str, session: requests.Session) -> str:
    _limiter.wait()          # <-- always called before the HTTP request
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.text
```

The `_limiter` is a module-level singleton — shared across all calls in the
same process, ensuring delays are enforced globally.

---

## 5. Retry Logic and Exponential Backoff

### Why retry?

Web servers return transient errors: `503 Service Unavailable`, connection
timeouts, rate-limit 429s. A single failure should not abort the entire scrape.

### tenacity `@retry` decorator

```python
# src/scrapers/motley_fool.py
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),                        # give up after 3 tries
    wait=wait_exponential(multiplier=2, min=4, max=30), # 4s, 8s, 16s, capped at 30s
    reraise=True,                                       # re-raise the last exception
)
def fetch_page(url: str, session: requests.Session) -> str:
    ...
```

### What is exponential backoff?

Each retry waits longer than the previous. With `multiplier=2, min=4`:
- Attempt 1 fails → wait 4 seconds
- Attempt 2 fails → wait 8 seconds
- Attempt 3 fails → wait 16 seconds (capped at 30) → `reraise=True` propagates
  the exception to the caller

Why exponential (not linear)? A server overloaded enough to return 503 won't
recover in 4 seconds. Exponential growth gives it real time to recover.

### `reraise=True`

Without this, `tenacity` would swallow the exception after all retries and
return `None`. With `reraise=True`, the original `requests.HTTPError` propagates
to `run_scraper()` which catches it and logs it:

```python
except Exception as exc:
    logger.error("Failed to scrape %s %s (%s): %s", ticker, date, url, exc)
    failed += 1
```

---

## 6. Manifest-Driven Scraping

### Why pre-curate URLs instead of crawling?

Crawling (discovering URLs from a site map) has several problems:
1. Requires parsing navigation pages — fragile, breaks on site redesigns
2. Can inadvertently violate `robots.txt` if discovery pages are blocked
3. Produces non-reproducible datasets (new articles added daily)

The manifest approach: commit `data/raw/url_manifest.csv` to git with exactly
192 pre-curated URLs. Every run of the scraper produces the same dataset.

```csv
# data/raw/url_manifest.csv (first few lines)
ticker,company,date,url,sector
AAPL,Apple Inc.,2022-01-27,https://www.fool.com/earnings/call-transcripts/.../,Technology
MSFT,Microsoft Corporation,2022-01-25,...,Technology
```

### Loading the manifest

```python
# src/scrapers/url_builder.py
def load_manifest(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
```

`csv.DictReader` reads the CSV header row and returns each subsequent row as a
`dict` keyed by column name. No pandas needed for a simple lookup.

### Idempotency (safe to re-run)

```python
# src/scrapers/motley_fool.py
def _already_scraped(ticker: str, date: str) -> bool:
    path = RAW_DIR / f"{ticker.upper()}_{date}.json"
    return path.exists()
```

If the scraper is interrupted after 100/192 transcripts, re-running it skips
the 100 already on disk. This is idempotency — multiple runs produce the same
end state.

---

## 7. HTML Parsing with BeautifulSoup

### What does BeautifulSoup do?

HTML is not plain text. A transcript page contains headers, footers, navigation,
ads, and the article body. BeautifulSoup builds a parse tree from the raw HTML
string so we can navigate to specific elements by tag name or CSS class.

```python
# src/scrapers/motley_fool.py
def parse_transcript(html: str, url: str, ticker: str, date: str, company: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # Primary selector: Motley Fool wraps article text in <div class="article-body">
    body = soup.find("div", class_="article-body")
    if body is None:
        body = soup.find("div", {"id": "fool-article-body"})
    if body is None:
        body = soup  # last resort: search the whole page

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
```

### Why the three-level fallback?

Motley Fool has redesigned its layout multiple times. Over 192 transcripts
spanning 2022–2023, some pages use `class="article-body"`, some use an `id`
attribute, and a few edge cases have neither. The three-level fallback ensures we
extract something useful even on older or reformatted pages.

### `get_text(separator=" ", strip=True)`

- `separator=" "` inserts a space between inline elements, preventing words
  from two adjacent `<span>` tags merging into one ("revenueincreased").
- `strip=True` removes leading/trailing whitespace from each paragraph.

### `lxml` parser vs. `html.parser`

The `"lxml"` argument selects the backend parser. `lxml` is C-based and faster
than Python's built-in `html.parser`. It also handles malformed HTML more
gracefully (common on real scraped pages).

---

## 8. Regex-Based Transcript Segmentation

This is the central scientific artifact of the project. The quality of the
Prepared Remarks / Q&A split directly determines the validity of the 4-way
comparison.

### The three-tier strategy

```
raw_text
   │
   ▼
Tier 1: Does "Prepared Remarks" appear before a Q&A header?
   YES → split_method="regex_primary", split_successful=True
   │
   ▼
Tier 2: Does an "Operator:" speaker turn appear after the first 1/3 of text?
   YES → split_method="regex_fallback", split_successful=True
   │
   ▼
Tier 3: Heuristic 60% split
         split_method="heuristic", split_successful=False
         → EXCLUDED from evaluation metrics
         → LOGGED as failure case (type: split_heuristic)
```

### The compiled patterns

```python
# src/processing/segmenter.py
_PREPARED_PATTERN = re.compile(
    r"prepared\s+remarks",
    re.IGNORECASE,
)

_QA_PATTERN = re.compile(
    r"(?:"
    r"questions?\s+(?:and\s+)?answers?"          # "Questions and Answers"
    r"|q\s*(?:and|&)\s*a\s*(?:session)?"          # "Q&A" or "Q and A Session"
    r"|question[- ]and[- ]answer"                  # "Question-and-Answer"
    r"|open\s+(?:the\s+)?(?:floor|lines?)\s+for\s+questions?"  # "open the floor for questions"
    r"|we\s+(?:will\s+)?(?:now\s+)?(?:begin|open|take)\s+(?:the\s+)?q(?:uestion)?s?"
    r")",
    re.IGNORECASE,
)

_OPERATOR_TURN = re.compile(
    r"(?:^|\n)Operator\s*(?:\n|:)",
    re.IGNORECASE,
)
```

### Regex anatomy

**`re.IGNORECASE`** — makes the match case-insensitive. "Questions and Answers",
"QUESTIONS AND ANSWERS", and "questions and answers" all match `_QA_PATTERN`.

**`\s+`** — matches one or more whitespace characters (space, tab, newline).
"prepared remarks" and "prepared  remarks" (double space) both match.

**`(?:...)`** — a non-capturing group. Groups alternatives without creating a
numbered capture group. `questions?` matches "question" or "questions" (`?`
makes the final `s` optional).

**`|`** — alternation. Matches any one of the listed alternatives left-to-right.

**`(?:^|\n)`** in `_OPERATOR_TURN` — matches either the very start of the string
or a newline. This ensures "Operator" is at the beginning of a line, not
inside running text like "The operator confirmed that...".

**`(?:\n|:)`** — matches either a newline or a colon after "Operator". Motley
Fool uses two formats across different transcript eras: old transcripts write
`Operator: Good day...` (colon), while newer transcripts write the speaker name
on its own line (`Operator\nGood day...`). The alternation handles both.

**Why Tier 2 skips the first Operator turn:** In the new Motley Fool format,
"Operator" appears at position ~0 to open the call. Tier 2 only considers
Operator turns that appear after the first 1/3 of the text — this skips the
opening turn and finds the Operator returning to begin analyst Q&A.

**NFLX and TSLA exception:** These companies use a non-standard interview/webcast
format with no Operator at all (NFLX uses an external interviewer; TSLA calls
itself a "Q&A webcast"). All 13 transcripts from these two tickers fall to the
Tier 3 heuristic split and are excluded from evaluation metrics, but they
serve as failure-case examples in the error analysis.

### The split logic

```python
# src/processing/segmenter.py
def split_transcript(raw_text: str, ticker: str, date: str) -> TranscriptSegments:

    # Tier 1
    pr_match = _PREPARED_PATTERN.search(raw_text)
    qa_match = _QA_PATTERN.search(raw_text)

    if pr_match and qa_match and pr_match.start() < qa_match.start():
        prepared = raw_text[pr_match.end(): qa_match.start()].strip()
        qa = raw_text[qa_match.end():].strip()
        return TranscriptSegments(..., split_successful=True, split_method="regex_primary")

    # Tier 2
    midpoint = len(raw_text) // 3
    for m in _OPERATOR_TURN.finditer(raw_text):
        if m.start() > midpoint:
            prepared = raw_text[:m.start()].strip()
            qa = raw_text[m.start():].strip()
            return TranscriptSegments(..., split_successful=True, split_method="regex_fallback")

    # Tier 3
    split_idx = int(len(raw_text) * 0.60)
    return TranscriptSegments(..., split_successful=False, split_method="heuristic")
```

**Why `pr_match.start() < qa_match.start()`?** This guards against transcripts
where the regex matches a stray mention of "Q&A" before "Prepared Remarks" in
the header or preamble. We require the structural order to be correct: PR header
comes first, then Q&A header.

**Why discard Tier 3 from evaluation?** A 60% split is arbitrary — we have no
evidence the boundary is correct. Including these rows would add noise to the
correlation metrics without a valid sentiment signal.

---

## 9. Text Cleaning Pipeline

```python
# src/processing/cleaner.py (simplified view of the pipeline)
def clean_text(text: str) -> str:
    text = remove_speaker_labels(text)   # "Tim Cook -- CEO:" → ""
    text = remove_boilerplate(text)      # legal disclaimers, forward-looking statements header
    text = normalize_whitespace(text)    # collapse multiple spaces/newlines
    return text
```

### Speaker labels

Earnings call transcripts include speaker labels before every turn:
```
Tim Cook -- CEO: Thank you. Good afternoon everyone.
Luca Maestri -- CFO: Revenue was $123 billion...
```

These labels carry no sentiment signal but they do contain words like "CEO" which
could confuse the LM lexicon. `remove_speaker_labels()` strips lines matching
the pattern `Name -- Role:` before tokenization.

### Why NOT remove stop words in the cleaner?

This was an intentional design decision. Stop words are removed in
`tokenize_for_lexicon()` (before LM scoring), but the cleaner outputs text that
FinBERT also reads.

**FinBERT requires full sentences.** The sentence "We did not see revenue growth"
becomes "We did not see revenue growth" — removing stop words produces "revenue
growth", which has completely the opposite sentiment. BERT's attention mechanism
needs grammatical context words to understand negation, hedging ("could",
"might"), and modality ("expected", "anticipated").

---

## 10. Tokenization — LM vs. BERT

### Two completely different tokenization pipelines

| Property | LM Lexicon | FinBERT |
|---|---|---|
| Library | NLTK punkt | HuggingFace tokenizer |
| Stop words | REMOVED | KEPT |
| Case | UPPERCASE | Original (BERT-cased) |
| Output | `list[str]` | token IDs (`list[int]`) |
| Why | Dictionary lookup | Neural attention |

### LM tokenization

```python
# src/processing/tokenizer.py
def tokenize_for_lexicon(text: str, remove_stops: bool = True) -> list[str]:
    text = text.lower().translate(_PUNCT_TABLE)   # 1. lowercase + strip punctuation
    tokens = word_tokenize(text)                   # 2. NLTK punkt tokenizer
    if remove_stops:
        tokens = [t for t in tokens if t not in _ENGLISH_STOPWORDS]  # 3. drop stops
    return [t.upper() for t in tokens if len(t) >= 2 and t.isalpha()]  # 4. UPPERCASE
```

The final list looks like: `["REVENUE", "GROWTH", "STRONG", "POSITIVE", ...]`

**Why UPPERCASE?** The LM Master Dictionary stores all words in uppercase
(`"Word"` column contains "STRONG", "RISK", etc.). The lookup `t in lm["positive"]`
requires the token case to match the dictionary exactly.

**Why `t.isalpha()`?** Filters out numbers, currency symbols, and punctuation
remnants. "Q3", "$500M", "2023" are not in the LM dictionary.

### BERT tokenization (chunking)

```python
# src/processing/tokenizer.py
def chunk_for_bert(text: str, tokenizer, max_tokens: int = 512, stride: int = 128) -> list[str]:
    effective_max = max_tokens - 2  # reserve slots for [CLS] and [SEP]

    encoding = tokenizer(text, add_special_tokens=False, truncation=False)
    token_ids: list[int] = encoding["input_ids"]

    if len(token_ids) <= effective_max:
        return [text]  # short enough — no chunking needed

    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + effective_max, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(token_ids):
            break
        start += stride   # advance by stride (not by effective_max)

    return chunks
```

The output is a list of *text strings* (not token IDs) because the HuggingFace
`pipeline` API expects strings. We tokenize only to count tokens and split;
then decode each chunk back to text.

---

## 11. Market-Adjusted Returns (Ground Truth)

### What are "adjusted close" prices?

Raw stock prices are split-adjusted and dividend-adjusted. When Apple does a
4-for-1 split, each share price is multiplied by 0.25 in historical records.
This ensures returns computed over any window are economically meaningful.

`yfinance` with `auto_adjust=True` downloads the adjusted close price directly.

### What does "market-adjusted" mean?

```python
# src/processing/returns.py
market_adj_1d = stock_return_1d - spy_return_1d
```

On a day when the entire market rose 1% (SPY = +1%), a stock that also rose 1%
beat no one — it moved with the market. Only the *excess* return (above SPY)
reflects the market's specific reaction to the earnings call.

### Computing 1-day return

```python
def _get_price_return(ticker: str, start: str, end: str) -> float | None:
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty or len(data) < 2:
        return None
    prices = data["Close"].squeeze()
    return float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0])
```

**`pd.offsets.BDay(1)`** — Business Day offset. If earnings were on a Friday,
`BDay(1)` gives the following Monday (not Saturday). This avoids computing a
"return" across a weekend when markets are closed.

**Critical detail — yfinance end date is exclusive.** A common pitfall:

```python
# WRONG — returns only 1 row (earnings date itself), len(data) < 2 → None
date_plus_1 = (date + pd.offsets.BDay(1)).strftime("%Y-%m-%d")
yf.download(ticker, start=date_str, end=date_plus_1)  # [earnings_date, next_bday)

# CORRECT — returns 2 rows: earnings date close AND next trading day close
date_plus_1_excl = (date + pd.offsets.BDay(2)).strftime("%Y-%m-%d")
yf.download(ticker, start=date_str, end=date_plus_1_excl)  # [earnings_date, bday+1)
```

The 1-day return is then `(prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]`,
i.e. the percentage change from the earnings-day close to the next trading day's
close — exactly what we want.

### Why might `data.empty` happen?

- The ticker was delisted or renamed in the data window
- The date is a US public holiday (no trading)
- yfinance network error or data gap

The `return None` guard propagates as `NaN` in the DataFrame and those rows
are excluded from Pearson correlation via `.dropna()`.

---

## 12. Loughran-McDonald Lexicon

### What is the LM Dictionary?

Published by Tim Loughran and Bill McDonald (Notre Dame), it is a financial
domain word list built from analyzing 10-K SEC filings from 1993–2023.
It corrects a well-known problem: General-purpose dictionaries (like Harvard IV)
classify words like "liability", "tax", "board", and "cancer" as negative —
but in financial filings these words are neutral.

The LM Dictionary contains ~86,000 words in five sentiment categories:
Positive, Negative, Uncertainty, Litigious, Constraining.

### Loading with `lru_cache`

```python
# src/models/lm_lexicon.py
@lru_cache(maxsize=1)
def load_lm_dictionary(csv_path: Path = LM_CSV_PATH) -> dict[str, set[str]]:
    df = pd.read_csv(csv_path, usecols=["Word"] + _CATEGORIES, low_memory=False)
    df["Word"] = df["Word"].str.upper().str.strip()

    result: dict[str, set[str]] = {}
    for col in _CATEGORIES:
        result[col.lower()] = set(df.loc[df[col] != 0, "Word"])

    return result
```

**`@lru_cache(maxsize=1)`** — Least Recently Used cache with a size limit of 1.
Because `maxsize=1`, the cache stores exactly one result: the first call reads
the CSV and caches it; all subsequent calls return the cached dict instantly.
Without this, reading a large CSV on every scoring call would be very slow.

**Why `set` instead of `list`?** Membership testing: `"REVENUE" in lm["positive"]`
is O(1) for a set and O(n) for a list. With 86,000 words and millions of token
lookups, the difference is large.

### The scoring formula

```python
# src/models/lm_lexicon.py
def compute_lm_score(tokens: list[str]) -> dict[str, float]:
    lm = load_lm_dictionary()
    n = len(tokens)

    pos = sum(1 for t in tokens if t in lm["positive"]) / n
    neg = sum(1 for t in tokens if t in lm["negative"]) / n

    return {
        "positive_ratio": round(pos, 6),
        "negative_ratio": round(neg, 6),
        "net_sentiment": round(pos - neg, 6),   # KEY FEATURE
        ...
        "n_tokens": n,
    }
```

**Normalized ratios** divide by `n` (token count) so that a 5,000-word prepared
remarks section and a 1,000-word Q&A section produce comparable scores. Raw
counts would systematically favor longer texts.

**net_sentiment** is the continuous feature used for Pearson correlation.
`classify_sentiment()` applies a threshold to convert it to a 3-class label:

```python
def classify_sentiment(scores: dict[str, float], threshold: float = 0.01) -> str:
    net = scores["net_sentiment"]
    if net > threshold:   return "positive"
    if net < -threshold:  return "negative"
    return "neutral"
```

**Why threshold = 0.01?** A net sentiment of 0.01 means the text has 1% more
positive financial words than negative ones. This is a weak signal — many
neutral-sounding earnings calls hit this level. The threshold is swept in
evaluation (Phase 5) to find the F1-optimal cutoff.

---

## 13. FinBERT: Architecture and Why It Exists

### What is BERT?

BERT (Bidirectional Encoder Representations from Transformers, Google 2018) is
a transformer-based language model. Key properties:

- **Pre-trained** on Wikipedia + BooksCorpus (billions of tokens)
- **Bidirectional**: reads the full sentence left *and* right simultaneously
  (unlike GPT which only reads left-to-right)
- **Contextual embeddings**: the same word gets a different vector depending on
  the surrounding context ("bank" in "river bank" ≠ "bank" in "bank account")

### What is FinBERT?

FinBERT (`ProsusAI/finbert`) is BERT fine-tuned on:
- Financial PhraseBank (annotated financial sentences)
- Financial news articles
- 10-K/10-Q SEC filings

**Why does fine-tuning matter?** BERT's original vocabulary and pre-training
data is general English. Financial language is highly domain-specific:
"headwinds", "guidance", "beat estimates", "operating leverage", "dilution".
Fine-tuning on financial text adjusts BERT's internal representations so these
terms receive appropriate contextual meaning.

**Output:** Three softmax probabilities — `positive`, `negative`, `neutral`.
These sum to 1.0 for any input. The predicted class is `argmax`.

### Loading FinBERT

```python
# src/models/finbert.py
def load_finbert_pipeline(device: int = -1):
    if device == -1 and torch.cuda.is_available():
        device = 0  # auto-upgrade to GPU if available

    pipe = hf_pipeline(
        "text-classification",
        model=MODEL_ID,            # "ProsusAI/finbert"
        tokenizer=MODEL_ID,
        device=device,
        top_k=None,                # return ALL three class scores
        truncation=True,           # safety truncation
        max_length=512,
    )
    return pipe
```

**`top_k=None`** is critical. By default, HuggingFace pipelines return only
the top-1 predicted class. With `top_k=None`, we get the full probability
distribution: `[{"label": "positive", "score": 0.82}, {"label": "neutral",
"score": 0.12}, {"label": "negative", "score": 0.06}]`. This allows us to
compute continuous scores (positive probability) for Pearson correlation.

### The 512-token limit

BERT uses fixed-length positional embeddings. Position 0, 1, 2, ... up to 511.
There are no embeddings defined for position 512+. Tokens beyond position 511
*cannot be processed* — the model simply has no representation for them.

A Q&A section of a 30-minute call can easily be 3,000–10,000 tokens. Without
chunking, we would truncate 80–90% of the Q&A text.

---

## 14. Sliding Window Chunking

### The problem

```
Full Q&A text:  10,000 tokens
BERT limit:     512 tokens (including 2 special tokens → 510 usable)
```

Naive truncation: take first 510 tokens, discard the rest. This means analyst
follow-up questions in the second half of Q&A — often where the most revealing
answers appear — are never scored.

### The sliding window solution

```
Token positions:  0    128   256   384   510  638  766  ...  9999
                  |←—chunk 1 (510)—→|
                       |←—chunk 2 (510)—→|
                            |←—chunk 3 (510)—→|
```

Each chunk overlaps with the previous by `510 - 128 = 382` tokens. Sentiment
signals near a chunk boundary appear in two overlapping chunks, so no signal
is captured by only one chunk.

```python
# src/processing/tokenizer.py
def chunk_for_bert(text: str, tokenizer, max_tokens: int = 512, stride: int = 128) -> list[str]:
    effective_max = max_tokens - 2  # 510

    encoding = tokenizer(text, add_special_tokens=False, truncation=False)
    token_ids = encoding["input_ids"]

    if len(token_ids) <= effective_max:
        return [text]

    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(start + effective_max, len(token_ids))
        chunk_ids = token_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(token_ids):
            break
        start += stride   # ← advance by stride, not effective_max

    return chunks
```

### Score aggregation

```python
# src/models/chunker.py
def aggregate_chunk_scores(
    chunk_scores: list[dict[str, float]],
    strategy: str = "mean"
) -> dict[str, float]:
    if not chunk_scores:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "label": "neutral"}

    labels = ["positive", "negative", "neutral"]
    if strategy == "mean":
        avg = {label: sum(s.get(label, 0.0) for s in chunk_scores) / len(chunk_scores)
               for label in labels}
        avg["label"] = max(avg, key=lambda k: avg[k] if k in labels else -1)
        return avg
    ...
```

**Mean aggregation:** Average the three class probabilities across all chunks.
This treats all parts of the text equally. It is the default.

**Why not max?** "Max" would give undue weight to the most extreme chunk,
making outlier paragraphs determine the score for the entire transcript.

---

## 15. Evaluation Metrics

### Accuracy

Fraction of predictions that exactly match the ground-truth label.

```
Accuracy = (TP_pos + TP_neu + TP_neg) / N
```

**Limitation in class-imbalanced settings:** If 60% of labels are "neutral",
a classifier that always predicts "neutral" achieves 60% accuracy without
learning anything. This is why we also report F1.

### Macro F1

F1 is the harmonic mean of Precision and Recall:
```
F1_class = 2 × (Precision × Recall) / (Precision + Recall)
Macro F1 = average of F1_pos, F1_neu, F1_neg (unweighted)
```

**Macro F1 treats all classes equally** regardless of how many samples each has.
It penalizes a model that ignores the minority class.

### Weighted F1

Same as macro but weights each class's F1 by its support (number of samples).
It matches accuracy in class-balanced settings and is more informative in
imbalanced settings than accuracy.

```python
# src/evaluation/metrics.py
def compute_classification_metrics(y_true, y_pred, labels=_LABELS):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "positive_f1": round(per_class[0], 4),
        "neutral_f1": round(per_class[1], 4),
        "negative_f1": round(per_class[2], 4),
    }
```

**`zero_division=0`**: If a class never appears in predictions, F1 is undefined
(division by zero). This flag returns 0 instead of raising a warning.

### Pearson Correlation (r)

Pearson r measures the *linear* relationship between two continuous variables.

```python
# src/evaluation/metrics.py
def compute_pearson_correlation(sentiment_scores, market_returns):
    r, p = pearsonr(sentiment_scores, market_returns)
    return (round(float(r), 4), round(float(p), 4))
```

For LM: `sentiment_scores = net_sentiment` (continuous, range roughly -0.05 to +0.05)
For FinBERT: `sentiment_scores = finbert_positive` (probability, range 0 to 1)

**r values:**
- r = +1.0: perfect positive linear relationship
- r = 0.0: no linear relationship
- r = -1.0: perfect inverse relationship

In financial NLP, r > 0.15 with p < 0.05 is considered meaningful.

**p-value:** The probability of observing r ≥ |this value| if the true
correlation were zero. p < 0.05 means we can reject the null hypothesis of no
correlation at the 5% significance level.

### The 4-way comparison

```python
# src/evaluation/metrics.py
def compare_models(df: pd.DataFrame, ground_truth_col: str = "label_1d") -> pd.DataFrame:
    eval_df = df[df["split_successful"]].dropna(subset=[ground_truth_col])
    y_true = eval_df[ground_truth_col].tolist()

    conditions = {
        "PR_LM":      "pr_lm_label",
        "QA_LM":      "qa_lm_label",
        "PR_FINBERT":  "pr_finbert_label",
        "QA_FINBERT":  "qa_finbert_label",
    }

    for condition, pred_col in conditions.items():
        y_pred = eval_df[pred_col].tolist()
        metrics = compute_classification_metrics(y_true, y_pred)
        # also compute Pearson r for 1d and 3d returns
        ...
```

**`df[df["split_successful"]]`** is the key filter. Heuristic-split rows are
excluded from metrics because their segment boundaries are unreliable, which
would add random noise to the evaluation.

---

## 16. Error Analysis

### Why mandatory?

The grading rubric requires ≥10 analyzed failure cases. This is standard
scientific practice: qualitative analysis of failures reveals systematic
weaknesses that aggregate metrics cannot show.

### Two types of failures

```python
# src/evaluation/error_analysis.py
def extract_failure_cases(df, model_label_col, ground_truth_col="label_1d", min_cases=10):

    # Type 1: Sentiment mismatch — model's prediction differs from market direction
    eval_df = df[df.get("split_successful", pd.Series([True]*len(df)))]
    mismatches = eval_df[eval_df[model_label_col] != eval_df[ground_truth_col]].copy()

    def _reason(row):
        pred, truth = row[model_label_col], row[ground_truth_col]
        if pred == "positive" and truth == "negative":  return "false_positive"
        if pred == "negative" and truth == "positive":  return "false_negative"
        return "neutral_missed"

    # Type 2: Structural failures — regex splitting didn't find section markers
    heuristic_df = df[df["split_method"] == "heuristic"].copy()
    heuristic_df["failure_reason"] = "split_heuristic"

    combined = pd.concat([mismatches, heuristic_df], ignore_index=True)
    ...
    if len(result) < min_cases:
        raise ValueError(f"Found only {len(result)} failure cases...")
```

**Type 1 reasons:**
- `false_positive`: Model predicted positive sentiment, but the stock fell.
  Common cause: management spoke optimistically about the wrong things (AI
  investments, product launches) while analysts were concerned about margins.
- `false_negative`: Model predicted negative, stock rose. Common cause: earnings
  beats were quantitative (numbers exceeded estimates) but language was cautious.
- `neutral_missed`: Model predicted neutral but market moved significantly.
  Common cause: LM threshold too high — a meaningful net_sentiment of 0.008
  is classified as neutral.

**Type 2 reason:**
- `split_heuristic`: The transcript's section structure was not recognized.
  This often happens with foreign company transcripts (ADRs), very short
  transcripts, or unusual Motley Fool formatting.

### The hard guard

```python
if len(result) < min_cases:
    raise ValueError(...)
```

This is a deliberate design choice. Silently producing a report with only 3
failure cases would pass without notice. Raising `ValueError` forces the researcher
(and the grader) to notice that the failure analysis requirement is not met.

---

## 17. Visualizations

Four plots are produced at 300 DPI (publication quality) in `data/outputs/figures/`:

### Plot 1 — Sentiment vs. Returns Scatter

Shows each transcript as a point. X-axis: continuous sentiment score
(net_sentiment for LM, positive probability for FinBERT). Y-axis: market-adjusted
1-day return. The Pearson r and p-value are annotated on the plot.

**How to read it:** A visible upward trend (positive slope) indicates higher
sentiment predicts higher returns. Scatter around the trend line shows how much
variance sentiment explains.

### Plot 2 — Model Comparison Bar Chart (F1)

Grouped bar chart with one bar per condition (PR_LM, QA_LM, PR_FINBERT, QA_FINBERT).
Shows macro F1 and weighted F1 side by side.

**How to read it:** If QA_FINBERT bars are consistently taller, Q&A + FinBERT
is the best-performing combination. The difference between macro and weighted F1
bars reveals class imbalance effects.

### Plot 3 — Sentiment Distribution

Stacked bar chart showing how many transcripts were labeled positive/neutral/negative
by each model × segment combination.

**Why this matters:** Class imbalance is visible here. If 70% of predictions
are "neutral", that explains why accuracy is high but macro F1 is low.

### Plot 4 — Pearson Correlation Heatmap

A 4×2 heatmap: rows are conditions (PR_LM, QA_LM, PR_FINBERT, QA_FINBERT),
columns are return windows (1d, 3d). Cell color represents r value.

**How to read it:** Warm colors (positive r) in the FinBERT rows but cool colors
in the LM rows would support the hypothesis that neural models extract more
predictive signal than bag-of-words lexicons.

---

## 18. End-to-End Pipeline Flow

### `python main.py --mode scrape`

```
main.py
  └─ src.scrapers.motley_fool.run_scraper()
       ├─ load_manifest()                     → list of 192 {ticker, date, url}
       ├─ _make_session()                     → requests.Session with User-Agent
       └─ for each entry:
            ├─ _already_scraped() → skip if JSON exists
            ├─ fetch_page()       → HTML string (with retry + rate limit)
            ├─ parse_transcript() → dict {ticker, date, company, raw_text, ...}
            └─ save_transcript()  → data/raw/transcripts/AAPL_2022-01-27.json
```

### `python main.py --mode analyze --model both`

```
main.py
  └─ src.pipeline.run_analysis()
       ├─ storage.list_transcripts()         → all .json paths
       ├─ [load each JSON → list of dicts]
       │
       ├─ PREPROCESSING
       │   ├─ cleaner.clean_text()           → cleaned raw_text
       │   ├─ segmenter.split_transcript()   → TranscriptSegments
       │   ├─ tokenizer.tokenize_for_lexicon() → prepared_tokens, qa_tokens
       │   └─ returns.build_returns_dataframe() → market_adj_1d, label_1d
       │       (cached to data/processed/returns.csv)
       │   → saved to data/processed/segments.parquet
       │
       ├─ LM SCORING (--model lm or both)
       │   ├─ lm_lexicon.score_dataframe(token_col='prepared_tokens', prefix='pr_lm')
       │   └─ lm_lexicon.score_dataframe(token_col='qa_tokens', prefix='qa_lm')
       │
       ├─ FINBERT SCORING (--model finbert or both)
       │   ├─ finbert.load_finbert_pipeline()
       │   ├─ finbert.score_dataframe(text_col='prepared_text', prefix='pr_finbert')
       │   │    └─ for each row: predict_sentiment() → chunk_for_bert() → pipeline()
       │   └─ finbert.score_dataframe(text_col='qa_text', prefix='qa_finbert')
       │        (cached to data/outputs/finbert_scores.parquet)
       │
       ├─ EVALUATION
       │   ├─ metrics.compare_models()        → results_summary.csv (4 conditions × metrics)
       │   └─ error_analysis.export_failure_cases() → failure_cases.csv (≥10 rows)
       │
       └─ VISUALIZATIONS
            ├─ plot_sentiment_vs_returns()    → figures/sentiment_vs_returns.png
            ├─ plot_model_comparison_f1()     → figures/model_comparison_f1.png
            ├─ plot_sentiment_distribution()  → figures/sentiment_distribution.png
            └─ plot_pearson_heatmap()         → figures/pearson_heatmap.png
```

### Data format: Parquet vs. CSV

`segments.parquet` uses Apache Parquet — a columnar binary format.

**Why Parquet instead of CSV?**
- **Stores list columns natively**: `prepared_tokens` is a `list[str]`. CSV would
  need to serialize the list as a string and parse it back — lossy and fragile.
  Parquet stores it as an actual list.
- **Faster reads**: Columnar format means reading only `prepared_tokens` doesn't
  load `raw_text` at all.
- **Type preservation**: Integer, float, bool, and datetime columns retain their
  types. CSV stores everything as strings and re-parsing is error-prone.

---

## 19. Key Design Decisions Q&A

**Q: What were the actual pipeline results?**
A: On 107 evaluable transcripts (124 scraped, 111 reliable splits, 4 dropped for
short segments):

| Condition | Accuracy | Macro F1 | Pearson r (1d) |
|---|---|---|---|
| PR_LM | 0.3925 | 0.1986 | 0.050 |
| QA_LM | 0.2150 | 0.1679 | -0.005 |
| **PR_FINBERT** | **0.4673** | **0.2538** | **0.058** |
| QA_FINBERT | 0.0748 | 0.0464 | -0.112 |

Key findings: Prepared Remarks + FinBERT is the best combination. QA_FINBERT
collapses to near-zero (predicts nearly all samples as the same class). All
Pearson correlations are statistically insignificant (p > 0.05). 78 failure
cases were exported (65 mismatches + 13 heuristic splits).

**Q: Why not use a Kaggle dataset instead of scraping?**
A: No suitable pre-existing dataset covers the 2022–2023 date range with both
earnings call text and contemporaneous stock returns. More importantly, building
the data pipeline is part of the academic exercise — scraping demonstrates
engineering competence.

**Q: Why is the URL manifest committed to git but transcripts are not?**
A: Transcripts are large (each JSON ~100KB, total ~20MB) and copyrighted. The
manifest is 124 lines of metadata — small, not copyrighted, and necessary for
reproducibility. Anyone running `python main.py --mode scrape` with the committed
manifest produces the exact same dataset.

**Q: Why use FinBERT instead of GPT-4 or a larger LLM?**
A: (1) FinBERT is specialized for financial text. (2) It runs locally without
an API key or cost. (3) 200 transcripts × 2 segments = 400 inference calls —
even on CPU, FinBERT completes in 1-2 hours with caching. GPT-4 would cost ~$10
and require internet access. (4) The research question compares a lexicon baseline
to a transformer — FinBERT is the standard choice for this comparison in the
academic literature.

**Q: What is the `split_successful` flag and why does it matter?**
A: It is a boolean that is `True` for regex_primary and regex_fallback splits,
and `False` for heuristic splits. Heuristic splits are excluded from evaluation
metrics because their segment boundaries are arbitrary — including them would
add noise to the correlation between sentiment and returns. They are still logged
as failure cases to satisfy the error analysis requirement.

**Q: Why does the Pearson correlation use continuous scores instead of class labels?**
A: Class labels (positive/neutral/negative) lose information — the difference
between net_sentiment=0.001 and net_sentiment=0.04 is discarded when both become
"positive". Pearson correlation on continuous scores measures the full linear
relationship between sentiment magnitude and return magnitude.

**Q: Could you fine-tune FinBERT on this dataset?**
A: Yes — this is mentioned as a "possible innovation" in the plan. You would
use the `label_1d` column as a weak supervision signal and fine-tune the
classification head. The risk is that with 107 evaluable samples (after train/val
split, ~75 training examples), overfitting is near-certain. For the current project scope,
zero-shot FinBERT is the appropriate choice.

**Q: What is the ethical justification for scraping Motley Fool?**
A: (1) `robots.txt` was verified before scraping — the transcript pages are
not blocked. (2) We used random delays (3-8s) to avoid server load. (3) Transcripts
are stored locally and not redistributed. (4) Usage is for non-commercial
academic research only. This falls within academic fair use of publicly accessible
content.

**Q: Why ±0.5% threshold for the return label?**
A: This threshold excludes the bid-ask spread noise (typically 0.01-0.1% for
large caps) and very small post-earnings drifts that are statistically
indistinguishable from random. A tighter threshold (e.g., ±0.1%) creates a very
large "positive" class on bull market days; a wider threshold (e.g., ±2%)
creates a very large "neutral" class and discards most signal. ±0.5% is the
standard in the academic earnings surprise literature.

---

*End of Study Guide — good luck with the defense.*
