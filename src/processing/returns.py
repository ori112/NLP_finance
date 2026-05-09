"""Fetch and compute market-adjusted stock returns using yfinance."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_RETURNS_CACHE = Path("data/processed/returns.csv")
_SPY = "SPY"  # S&P 500 ETF used as market benchmark


def _get_price_return(ticker: str, start: str, end: str) -> float | None:
    """Fetch the cumulative return for a ticker between two dates.

    Args:
        ticker: Stock ticker symbol.
        start: Start date string YYYY-MM-DD (inclusive).
        end: End date string YYYY-MM-DD (exclusive for yfinance).

    Returns:
        Fractional return (e.g. 0.02 = +2%) or None if data unavailable.
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if data.empty or len(data) < 2:
            return None
        prices = data["Close"].squeeze()
        return float((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0])
    except Exception as exc:
        logger.warning("yfinance error for %s [%s → %s]: %s", ticker, start, end, exc)
        return None


def fetch_returns(
    ticker: str,
    earnings_date: str,
    window_days: int = 3,
) -> dict[str, float | None]:
    """Fetch market-adjusted 1-day and 3-day returns around an earnings date.

    Market adjustment subtracts the SPY (S&P 500) return for the same period,
    isolating the stock-specific reaction from broad market noise.

    Date windows:
      - 1-day: earnings_date to earnings_date + 1 trading day
      - 3-day: earnings_date to earnings_date + 3 calendar days

    Args:
        ticker: Stock ticker symbol.
        earnings_date: Date of the earnings call (YYYY-MM-DD).
        window_days: Number of calendar days for the multi-day window.

    Returns:
        Dict with keys: return_1d, return_3d, market_adj_1d, market_adj_3d.
        Values are None if data could not be fetched.
    """
    date = pd.Timestamp(earnings_date)
    # yfinance end date is exclusive, so we need BDay(2) as end to include
    # both the earnings-date close AND the next trading day's close in the window.
    date_plus_1_excl = (date + pd.offsets.BDay(2)).strftime("%Y-%m-%d")
    date_plus_3 = (date + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    date_str = date.strftime("%Y-%m-%d")

    r1 = _get_price_return(ticker, date_str, date_plus_1_excl)
    r3 = _get_price_return(ticker, date_str, date_plus_3)
    spy1 = _get_price_return(_SPY, date_str, date_plus_1_excl)
    spy3 = _get_price_return(_SPY, date_str, date_plus_3)

    def _adj(stock: float | None, market: float | None) -> float | None:
        if stock is None or market is None:
            return None
        return stock - market

    return {
        "return_1d": r1,
        "return_3d": r3,
        "market_adj_1d": _adj(r1, spy1),
        "market_adj_3d": _adj(r3, spy3),
    }


def label_from_return(market_adj_return: float, threshold: float = 0.0) -> str:
    """Convert a market-adjusted return to a binary direction label.

    Per the project proposal, the classification target is the direction of
    stock movement (up vs. down) relative to the market. Default threshold
    of 0 makes the split a clean sign comparison.

    Args:
        market_adj_return: Market-adjusted return (fractional, e.g. 0.012).
        threshold: Boundary above which a return counts as "up".

    Returns:
        "up" if market_adj_return > threshold else "down".
    """
    return "up" if market_adj_return > threshold else "down"


def build_returns_dataframe(
    transcript_records: list[dict],
    cache_path: Path = _RETURNS_CACHE,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Build a DataFrame of market-adjusted returns for all transcripts.

    Results are cached to CSV after the first run so repeated yfinance
    calls are avoided.

    Args:
        transcript_records: List of transcript dicts (must have 'ticker' and 'date').
        cache_path: Path to cache CSV. Loads from cache if it exists.
        threshold: Label threshold passed to label_from_return().

    Returns:
        DataFrame with columns: ticker, date, return_1d, return_3d,
        market_adj_1d, market_adj_3d, label_1d, label_3d.
    """
    if cache_path.exists():
        logger.info("Loading returns from cache: %s", cache_path)
        df = pd.read_csv(cache_path)
        # Re-derive labels from the numeric returns so any change in
        # label_from_return() takes effect on cached data without re-fetching.
        for col, key in [("label_1d", "market_adj_1d"), ("label_3d", "market_adj_3d")]:
            df[col] = df[key].apply(
                lambda v: label_from_return(v, threshold) if pd.notna(v) else None
            )
        return df

    rows = []
    for rec in transcript_records:
        ticker = rec["ticker"]
        date = rec["date"]
        logger.info("Fetching returns for %s %s …", ticker, date)
        ret = fetch_returns(ticker, date)
        row = {"ticker": ticker, "date": date, **ret}

        for col, key in [("label_1d", "market_adj_1d"), ("label_3d", "market_adj_3d")]:
            val = ret.get(key)
            row[col] = label_from_return(val, threshold) if val is not None else None

        rows.append(row)

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    logger.info("Returns cached to %s", cache_path)
    return df
