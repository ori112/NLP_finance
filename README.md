# Earnings Call Sentiment Analysis

**Course:** M.Sc. NLP — HIT | **Track:** 3 — Data-Driven

## Research Question

Which combination of **(segment × model)** best predicts market-adjusted stock returns?

| Segment | Model | Condition |
|---|---|---|
| Prepared Remarks | Loughran-McDonald Lexicon | PR_LM |
| Q&A Section | Loughran-McDonald Lexicon | QA_LM |
| Prepared Remarks | FinBERT | PR_FINBERT |
| Q&A Section | FinBERT | QA_FINBERT |

## Quickstart

```bash
# 1. Install all dependencies
uv sync --extra dev

# 2. Download the LM Master Dictionary (one-time manual step)
#    → https://sraf.nd.edu/loughranmcdonald-master-dictionary/
#    → save to: data/raw/lm_dictionary/Loughran-McDonald_MasterDictionary_1993-2023.csv

# 3. Scrape ~192 earnings call transcripts (~20 min, polite rate limiting)
python main.py --mode scrape

# 4. Run the full analysis (LM + FinBERT)
python main.py --mode analyze --model both

# 5. Run tests (fast, no model download)
uv run pytest -m "not slow"
```

## Project Structure

```
src/
├── scrapers/
│   ├── motley_fool.py      # Transcript scraper with retry and rate limiting
│   ├── rate_limiter.py     # Random jitter delay for polite crawling
│   └── url_builder.py      # URL construction and manifest loading
├── processing/
│   ├── segmenter.py        # Regex-based Prepared Remarks / Q&A splitter
│   ├── cleaner.py          # Text cleaning (speaker labels, boilerplate)
│   ├── tokenizer.py        # LM tokenization (stop words removed) + BERT chunking
│   ├── returns.py          # yfinance market-adjusted return computation
│   └── pipeline.py         # Preprocessing orchestrator → segments.parquet
├── models/
│   ├── lm_lexicon.py       # Loughran-McDonald lexicon scorer
│   ├── finbert.py          # FinBERT inference with sliding window
│   └── chunker.py          # Sliding window chunking for 512-token BERT limit
├── evaluation/
│   ├── metrics.py          # Accuracy, F1, Pearson correlation
│   ├── error_analysis.py   # Failure case extraction and export
│   └── visualizations.py   # 4 publication-quality figures
├── utils/
│   └── storage.py          # JSON transcript save/load helpers
└── pipeline.py             # Full analysis orchestrator (--mode analyze)

tests/                      # pytest unit tests for all modules
notebooks/
├── 01_exploration.ipynb    # EDA
└── 02_full_pipeline.ipynb  # End-to-end pipeline (primary submission artifact)
data/
├── raw/
│   ├── url_manifest.csv            # 192 pre-curated transcript URLs (committed)
│   └── lm_dictionary/              # LM Master Dictionary CSV (manual download)
├── processed/                      # segments.parquet (gitignored)
└── outputs/                        # results_summary.csv, failure_cases.csv, figures/ (gitignored)
```

## Data Sources

| Source | Purpose | License |
|---|---|---|
| Motley Fool | Earnings call transcripts | Public / Academic fair use |
| yfinance | Stock price returns | Open source (Yahoo Finance) |
| LM Master Dictionary | Financial sentiment lexicon | Academic use |

## Results Summary

*(Populated after running `python main.py --mode analyze` on live data)*

| Condition | Accuracy | Macro F1 | Pearson r (1d) |
|---|---|---|---|
| PR_LM | — | — | — |
| QA_LM | — | — | — |
| PR_FINBERT | — | — | — |
| QA_FINBERT | — | — | — |

## CLI Reference

```bash
python main.py --mode scrape [--tickers AAPL MSFT ...]
python main.py --mode analyze [--model lm|finbert|both]
uv run pytest -m "not slow"     # fast tests only
uv run pytest                   # all tests (requires model download)
```
