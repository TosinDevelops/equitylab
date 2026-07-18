from __future__ import annotations

import pandas as pd
import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def _normalize_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns.name = None

    data.index = pd.to_datetime(data.index)
    data.index.name = "date"

    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename_map)

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    if "adj_close" not in data.columns:
        data["adj_close"] = data["close"]

    return data[["open", "high", "low", "close", "adj_close", "volume"]].dropna()


def load_data(
    ticker: str,
    period: str = "1d",
    interval: str = "1m",
    start: str | None = None,
    end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV from Yahoo Finance.

    Daily bars with an explicit start/end use a local DuckDB cache by default
    (only missing date ranges hit the network). Pass use_cache=False to force
    a fresh download.
    """
    ticker = normalize_ticker(ticker)

    if (
        use_cache
        and interval == "1d"
        and start is not None
        and end is not None
    ):
        from equitylab.data.cache import load_cached_prices

        # yfinance end is exclusive; cache API is inclusive on both ends.
        end_inclusive = (pd.Timestamp(end) - pd.Timedelta(days=1)).date()
        start_d = pd.Timestamp(start).date()
        if start_d <= end_inclusive:
            return load_cached_prices(ticker, start_d, end_inclusive)

    download_kwargs: dict = {
        "tickers": ticker,
        "interval": interval,
        "auto_adjust": False,
        "progress": False,
    }
    if start is not None or end is not None:
        download_kwargs["start"] = start
        download_kwargs["end"] = end
    else:
        download_kwargs["period"] = period

    raw = yf.download(**download_kwargs)
    if raw.empty:
        raise ValueError(f"No data found for {ticker}")

    return _normalize_price_frame(raw)
