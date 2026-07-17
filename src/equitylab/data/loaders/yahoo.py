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
) -> pd.DataFrame:
    ticker = normalize_ticker(ticker)

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
