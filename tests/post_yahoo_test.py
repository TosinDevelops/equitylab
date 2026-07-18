from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equitylab.screening.post_yahoo import (
    ScreenConfig,
    _passes_screen,
    apply_screen,
    compute_metrics,
    rsi,
    sma,
)


def test_sma_matches_simple_average() -> None:
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(close, period=3)
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 2
    assert result.iloc[4] == 4


def test_rsi_is_nan_before_rolling_period_is_full() -> None:
    moves = [1.0, -0.5] * 15
    close = pd.Series(100.0 + np.cumsum(moves))
    result = rsi(close, period=14)
    assert result.iloc[:14].isna().all()
    assert pd.notna(result.iloc[14])


def test_rsi_matches_expected_values() -> None:
    close = pd.Series([10.0, 11.0, 10.5, 11.5, 11.0, 12.0])
    result = rsi(close, period=2)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert float(result.iloc[2]) == pytest.approx(66.66666666666666)
    assert float(result.iloc[3]) == pytest.approx(85.71428571428571)
    assert float(result.iloc[4]) == pytest.approx(54.54545454545455)
    assert float(result.iloc[5]) == pytest.approx(81.48148148148148)


def test_rsi_undefined_without_losses() -> None:
    close = pd.Series([100.0 + i for i in range(30)])
    assert pd.isna(rsi(close, period=14).iloc[-1])


def _synthetic_prices(
    *,
    bars: int = 220,
    last_close: float = 80.0,
    peak: float = 100.0,
    last_volume: float = 2_000_000,
    base_volume: float = 1_000_000,
) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=bars, freq="B")
    close = np.full(bars, 90.0)
    close[100] = peak
    close[-1] = last_close
    volume = np.full(bars, base_volume)
    volume[-1] = last_volume
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=index,
    )


def test_compute_metrics_drawdown_and_relative_volume() -> None:
    prices = _synthetic_prices(
        last_close=80.0,
        peak=100.0,
        last_volume=2_000_000,
        base_volume=1_000_000,
    )
    metrics = compute_metrics(prices)

    assert metrics["close"] == pytest.approx(80.0)
    assert metrics["drawdown_52w"] == pytest.approx(-0.20)
    expected_rel_vol = 2_000_000 / ((19 * 1_000_000 + 2_000_000) / 20)
    assert metrics["relative_volume_20"] == pytest.approx(expected_rel_vol)


def test_passes_screen_max_drawdown_allows_milder_pullbacks() -> None:
    config = ScreenConfig(max_drawdown_52w=-0.90, max_rsi=100.0, min_relative_volume=0.0)
    assert _passes_screen(
        {
            "drawdown_52w": -0.10,
            "rsi_14": 40.0,
            "relative_volume_20": 1.0,
            "distance_from_sma_200": 0.0,
        },
        config,
    )
    assert _passes_screen(
        {
            "drawdown_52w": -0.50,
            "rsi_14": 40.0,
            "relative_volume_20": 1.0,
            "distance_from_sma_200": 0.0,
        },
        config,
    )
    assert not _passes_screen(
        {
            "drawdown_52w": -0.95,
            "rsi_14": 40.0,
            "relative_volume_20": 1.0,
            "distance_from_sma_200": 0.0,
        },
        config,
    )


def test_apply_screen_sets_entry_signal() -> None:
    panel = pd.DataFrame(
        {
            "drawdown_52w": [-0.10, -0.95],
            "rsi_14": [35.0, 35.0],
            "relative_volume_20": [1.5, 1.5],
            "distance_from_sma_200": [0.0, 0.0],
        },
        index=["GOOD", "BAD"],
    )
    config = ScreenConfig(max_drawdown_52w=-0.90, max_rsi=40.0, min_relative_volume=1.2)
    out = apply_screen(panel, config)

    assert bool(out.loc["GOOD", "entry_signal"]) is True
    assert bool(out.loc["BAD", "entry_signal"]) is False
    assert "signal_score" in out.columns


def test_screen_config_label_uses_positive_drawdown_percent() -> None:
    label = ScreenConfig(max_drawdown_52w=-0.30).label
    assert "30%" in label
    assert "RSI <= 40" in label
