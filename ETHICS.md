# Ethical and Responsible NLP Statement

> **Instructions:** Address each section below. This is a required submission artifact. Delete this instruction block before submission.

---

## Data Source Transparency

[TODO: Describe where the data comes from (Motley Fool, yfinance, LM Dictionary). State that transcript URLs are publicly accessible and that the data is used for non-commercial academic research only. No transcripts are redistributed.]

## Web Scraping Compliance

[TODO: State that you verified robots.txt before scraping, used randomised delays (3-8 seconds) to avoid server load, and did not scrape from institutional networks. Cite the specific rate-limiting implementation in src/scrapers/rate_limiter.py.]

## Privacy Considerations

[TODO: Earnings call transcripts are public corporate communications. No personal data (names, addresses, private communications) is collected or stored. Executive names appear as public figures in a professional context.]

## Bias in Data and Model Limitations

[TODO: Acknowledge that the LM dictionary was built from historical financial filings (1993-2023) and may not reflect modern language. FinBERT was trained on a specific corpus and may underperform on unusual phrasing. Market returns during 2022-2024 include significant macro events (rate hikes, geopolitical risk) that confound sentiment analysis.]

## Potential Harmful Applications

[TODO: State clearly that model outputs are NOT investment advice. Sentiment analysis of earnings calls should not be used as the basis for trading decisions. The project is for academic research purposes only.]

## Data Licensing and Copyright

[TODO: Motley Fool content is copyrighted. Usage is under academic fair use. The LM Master Dictionary is available for academic use. yfinance wraps the Yahoo Finance public API under its terms of service.]
