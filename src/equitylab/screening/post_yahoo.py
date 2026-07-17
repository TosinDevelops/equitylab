from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from equitylab.data.loaders.yahoo import load_data, normalize_ticker


@dataclass(frozen=True)
class ScreenConfig:
    max_drawdown_52w: float = -0.90
    max_rsi: float = 40.0
    min_relative_volume: float = 1.2
    min_distance_from_sma_200: float | None = None
    max_distance_from_sma_200: float | None = None

    @property
    def label(self) -> str:
        parts = [
            f"max 52w drawdown <= {abs(self.max_drawdown_52w):.0%}",
            f"RSI <= {self.max_rsi:.0f}",
            f"relative volume >= {self.min_relative_volume:.1f}",
        ]
        if self.min_distance_from_sma_200 is not None:
            parts.append(f"price vs 200D SMA >= {self.min_distance_from_sma_200:.0%}")
        if self.max_distance_from_sma_200 is not None:
            parts.append(f"price vs 200D SMA <= {self.max_distance_from_sma_200:.0%}")
        return ", ".join(parts)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(period, min_periods=period).mean()


def compute_metrics(prices: pd.DataFrame) -> dict[str, float]:
    close = prices["close"]
    volume = prices["volume"]

    high_52w = close.tail(252).max()
    drawdown_52w = float(close.iloc[-1] / high_52w - 1.0) if high_52w else float("nan")

    rsi_series = rsi(close, 14)
    sma_200 = sma(close, 200)
    vol_sma_20 = volume.rolling(20, min_periods=20).mean()

    last_close = float(close.iloc[-1])
    last_sma = float(sma_200.iloc[-1]) if pd.notna(sma_200.iloc[-1]) else float("nan")
    last_vol = float(volume.iloc[-1])
    last_vol_sma = float(vol_sma_20.iloc[-1]) if pd.notna(vol_sma_20.iloc[-1]) else float("nan")

    return {
        "close": last_close,
        "drawdown_52w": drawdown_52w,
        "rsi_14": float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else float("nan"),
        "relative_volume_20": last_vol / last_vol_sma if last_vol_sma else float("nan"),
        "distance_from_sma_200": last_close / last_sma - 1.0 if last_sma else float("nan"),
        "avg_volume_20": last_vol_sma,
    }


def apply_screen(feature_panel: pd.DataFrame, config: ScreenConfig) -> pd.DataFrame:
    data = feature_panel.copy()
    signal = (
        (data["drawdown_52w"] >= config.max_drawdown_52w)
        & (data["rsi_14"] <= config.max_rsi)
        & (data["relative_volume_20"] >= config.min_relative_volume)
    )
    if config.min_distance_from_sma_200 is not None:
        signal = signal & (
            data["distance_from_sma_200"] >= config.min_distance_from_sma_200
        )
    if config.max_distance_from_sma_200 is not None:
        signal = signal & (
            data["distance_from_sma_200"] <= config.max_distance_from_sma_200
        )

    data["entry_signal"] = signal.fillna(False)
    data["signal_score"] = _signal_score(data)
    return data


def _signal_score(data: pd.DataFrame) -> pd.Series:
    """Cross-sectional conviction score using latest metrics."""
    drawdown_score = data["drawdown_52w"].rank(pct=True, ascending=False)
    rsi_score = data["rsi_14"].rank(pct=True, ascending=False)
    volume_score = data["relative_volume_20"].rank(pct=True, ascending=True)
    return pd.concat([drawdown_score, rsi_score, volume_score], axis=1).mean(axis=1)


def _passes_screen(metrics: dict[str, float], config: ScreenConfig) -> bool:
    if not (metrics["drawdown_52w"] >= config.max_drawdown_52w):
        return False
    if not (metrics["rsi_14"] <= config.max_rsi):
        return False
    if not (metrics["relative_volume_20"] >= config.min_relative_volume):
        return False
    if config.min_distance_from_sma_200 is not None:
        if not (metrics["distance_from_sma_200"] >= config.min_distance_from_sma_200):
            return False
    if config.max_distance_from_sma_200 is not None:
        if not (metrics["distance_from_sma_200"] <= config.max_distance_from_sma_200):
            return False
    return True


def score_quotes(
    quotes: list[dict],
    screen: ScreenConfig,
    start: date,
    end: date,
    max_qualifiers: int = 50,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Score Yahoo quotes in order; stop once max_qualifiers pass ScreenConfig."""
    lookback_start = min(start, end - timedelta(days=400))
    rows: list[dict] = []
    errors: list[str] = []
    total = len(quotes)
    qualifiers = 0

    for index, quote in enumerate(quotes):
        symbol = normalize_ticker(str(quote.get("symbol", "")))
        if not symbol:
            continue
        if progress is not None:
            progress((index + 1) / total, f"Scoring {symbol} ({qualifiers}/{max_qualifiers} qualified)")

        try:
            prices = load_data(
                symbol,
                interval="1d",
                start=lookback_start.isoformat(),
                end=end.isoformat(),
            )
            if len(prices) < 200:
                errors.append(f"{symbol}: not enough history ({len(prices)} bars)")
                continue
            metrics = compute_metrics(prices)
            rows.append(
                {
                    "ticker": symbol,
                    "name": quote.get("shortName") or quote.get("longName") or symbol,
                    "market_cap": quote.get("marketCap"),
                    "avg_daily_volume_3m": quote.get("averageDailyVolume3Month"),
                    "universe_rank": index,
                    **metrics,
                }
            )
            if _passes_screen(metrics, screen):
                qualifiers += 1
                if qualifiers >= max_qualifiers:
                    break
        except Exception as exc:  # noqa: BLE001 - collect per-ticker failures
            errors.append(f"{symbol}: {exc}")

    if not rows:
        return pd.DataFrame(), errors or ["No tickers could be scored."]

    panel = pd.DataFrame(rows).set_index("ticker")
    scored = apply_screen(panel, screen)
    return scored, errors
