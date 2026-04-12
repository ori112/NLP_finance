# Dataset Documentation

> **Instructions:** Fill in the actual numbers after running the scraper. This file must be included in the ZIP submission. Delete this instruction block before submission.

---

## Primary Dataset: Earnings Call Transcripts

**Source:** Motley Fool — Earnings Call Transcripts  
**URL pattern:** `https://www.fool.com/earnings/call-transcripts/{year}/{month}/{day}/{slug}/`  
**Access method:** Web scraping via `src/scrapers/motley_fool.py` using the pre-curated manifest at `data/raw/url_manifest.csv`

| Attribute | Value |
|---|---|
| Total transcripts collected | 124 |
| Successful regex splits (PR/QA labelled) | 111 (7 primary + 104 Operator-turn fallback) |
| Heuristic fallback splits (excluded from metrics) | 13 (NFLX × 7, TSLA × 6 — interview/webcast format, no Operator) |
| Date range | Jan 2022 – Dec 2023 |
| Tickers covered | AAPL, AMZN, BA, CRM, CVX, DIS, GOOGL, INTC, KO, MA, MSFT, NFLX, NVDA, ORCL, PFE, SNAP, TSLA, V, WFC, WMT, XOM |
| Sectors covered | Technology, Financials, Healthcare, Energy, Consumer Discretionary, Consumer Staples, Industrials, Communication Services |

**Why this date range (2022–2023):**  
This window covers the most macroeconomically volatile and linguistically diverse period in recent history — the Fed's fastest rate hike cycle in 40 years, the Russia-Ukraine energy shock, the SVB banking crisis, and the emergence of the AI boom. Sentiment variation across this period is maximal in both directions, minimising class imbalance and making the Prepared Remarks vs. Q&A contrast particularly pronounced.

**Reproduction steps:**
```bash
# 1. Ensure data/raw/url_manifest.csv is present (committed to git)
# 2. Run the scraper (~20 minutes at 3-8s per request)
python main.py --mode scrape
# 3. Run preprocessing to produce segments.parquet
python main.py --mode analyze --model lm
```

**License / Terms:** Motley Fool content is copyrighted. This dataset is used solely for non-commercial academic research under fair use. Transcripts are stored locally and are not redistributed.

---

## Secondary Dataset: Stock Price Returns

**Source:** Yahoo Finance via the `yfinance` Python library  
**Purpose:** Compute 1-day and 3-day market-adjusted returns as ground-truth labels  
**Benchmark:** SPY (S&P 500 ETF) used as market return to compute excess return  
**License:** Yahoo Finance public API; yfinance is open source (Apache 2.0)

---

## Loughran-McDonald Financial Sentiment Dictionary

**Source:** University of Notre Dame — Tim Loughran & Bill McDonald  
**Download URL:** `https://sraf.nd.edu/loughranmcdonald-master-dictionary/`  
**File:** `Loughran-McDonald_MasterDictionary_1993-2025.csv`  
**Save to:** `data/raw/lm_dictionary/` (not committed to git — manual download required)  
**License:** Available for academic research use
