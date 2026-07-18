from __future__ import annotations

import pandas as pd

from equitylab.screening.post_yahoo import rsi, sma

FEATURE_COLUMNS = [
    "drawdown_52w",
    "rsi_14",
    "relative_volume_20",
    "distance_from_sma_200",
    "return_5d",
    "return_20d",
    "volatility_20d",
]


def build_feature_frame(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Build a daily feature DataFrame for one ticker from OHLCV prices."""
    if prices.empty:
        return pd.DataFrame(columns=["ticker", *FEATURE_COLUMNS])

    close = prices["close"].astype(float)
    volume = prices["volume"].astype(float)

    high_52w = close.rolling(252, min_periods=252).max()
    sma_200 = sma(close, 200)
    vol_sma_20 = volume.rolling(20, min_periods=20).mean()

    frame = pd.DataFrame(index=prices.index)
    frame["ticker"] = ticker
    frame["drawdown_52w"] = close / high_52w - 1.0
    frame["rsi_14"] = rsi(close, 14)
    frame["relative_volume_20"] = volume / vol_sma_20.replace(0, pd.NA)
    frame["distance_from_sma_200"] = close / sma_200 - 1.0
    frame["return_5d"] = close.pct_change(5)
    frame["return_20d"] = close.pct_change(20)
    frame["volatility_20d"] = close.pct_change().rolling(20, min_periods=20).std()
    frame["close"] = close
    frame["open"] = prices["open"].astype(float)
    frame["high"] = prices["high"].astype(float)
    frame["low"] = prices["low"].astype(float)
    frame["volume"] = volume
    frame.index.name = "date"
    return frame


def build_feature_panel(price_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-ticker feature frames into a MultiIndex (date, ticker) panel."""
    frames: list[pd.DataFrame] = []
    for ticker, prices in price_map.items():
        frame = build_feature_frame(prices, ticker)
        if frame.empty:
            continue
        frames.append(frame.reset_index())

    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", *FEATURE_COLUMNS, "close", "open", "high", "low", "volume"]
        ).set_index(["date", "ticker"])

    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.set_index(["date", "ticker"]).sort_index()
