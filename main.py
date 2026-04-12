import argparse
import logging
import sys


def _setup_logging() -> None:
    """Configure root logger to print timestamped messages to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Earnings Call Sentiment Analysis — Prepared Remarks vs. Q&A"
    )
    parser.add_argument(
        "--mode",
        choices=["scrape", "analyze"],
        required=True,
        help="scrape: collect transcripts from Motley Fool | analyze: run sentiment pipeline",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional list of ticker symbols to scope the scrape (e.g. --tickers AAPL MSFT)",
    )
    parser.add_argument(
        "--model",
        choices=["lm", "finbert", "both"],
        default="both",
        help="Which model(s) to run in analyze mode (default: both)",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()

    if args.mode == "scrape":
        from src.scrapers.motley_fool import run_scraper
        run_scraper(tickers=args.tickers)

    elif args.mode == "analyze":
        from src.pipeline import run_analysis
        run_analysis(model=args.model)


if __name__ == "__main__":
    main()
