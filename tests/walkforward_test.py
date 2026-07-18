from __future__ import annotations

import numpy as np
import pandas as pd

from equitylab.signals.features import FEATURE_COLUMNS, build_feature_panel
from equitylab.signals.labels import attach_labels
from equitylab.strategy.config import StrategyConfig
from equitylab.strategy.pipeline import run_walkforward_strategy
from equitylab.strategy.walkforward import (
    chronological_split_dates,
    fit_predict_walkforward,
    label_embargo_cutoff,
)


def _synthetic_ticker(
    ticker: str,
    bars: int = 400,
    seed: int = 0,
    drift: float = 0.0005,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2019-01-01", periods=bars, freq="B")
    noise = rng.normal(0, 0.01, size=bars)
    rets = drift + noise
    for i in range(50, bars - 30, 40):
        rets[i : i + 5] -= 0.02
        rets[i + 5 : i + 15] += 0.015
    close = 100.0 * np.cumprod(1.0 + rets)
    volume = rng.integers(800_000, 1_500_000, size=bars).astype(float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=index,
    )


def test_chronological_split_is_ordered() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    train_end, test_start = chronological_split_dates(dates, 0.8)
    assert train_end < test_start
    assert train_end == dates[7]
    assert test_start == dates[8]


def test_label_embargo_keeps_forward_horizon_inside_train() -> None:
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    train_end = dates[15]
    # N=5 → last train feature date must be dates[10] so label uses dates[15]
    cutoff = label_embargo_cutoff(dates, train_end, max_holding_days=5)
    assert cutoff == dates[10]


def test_fit_predict_embargo_excludes_overlapping_labels() -> None:
    price_map = {
        "AAA": _synthetic_ticker("AAA", seed=1),
        "BBB": _synthetic_ticker("BBB", seed=2),
        "CCC": _synthetic_ticker("CCC", seed=3),
    }
    panel = build_feature_panel(price_map)
    start = panel.index.get_level_values("date").min() + pd.Timedelta(days=400)
    end = panel.index.get_level_values("date").max()
    dates = panel.index.get_level_values("date")
    panel = panel.loc[(dates >= start) & (dates <= end)]
    hold = 5
    panel = attach_labels(panel, max_holding_days=hold)

    config = StrategyConfig(
        max_holding_days=hold,
        train_fraction=0.8,
        entry_min_return=0.0,
        profit_drawdown=0.05,
        max_positions=2,
    )
    wf = fit_predict_walkforward(panel, config)

    unique = pd.DatetimeIndex(sorted(panel.index.get_level_values("date").unique()))
    train_dates = unique[unique <= wf.train_end]
    cutoff = label_embargo_cutoff(unique, wf.train_end, hold)
    # Rows used for IS signals/preds are only through cutoff (see scored train.index)
    scored_train = wf.panel.loc[wf.panel["entry_signal_is"] | wf.panel["predicted_return"].notna()]
    scored_train = scored_train.loc[
        scored_train.index.get_level_values("date") <= wf.train_end
    ]
    max_train_feature_date = scored_train.index.get_level_values("date").max()
    assert max_train_feature_date <= cutoff
    # Label at cutoff ends exactly on train_end in trading-day space
    cutoff_pos = train_dates.get_loc(cutoff)
    assert train_dates[cutoff_pos + hold] == wf.train_end


def test_fit_predict_walkforward_predicts_forward_return() -> None:
    price_map = {
        "AAA": _synthetic_ticker("AAA", seed=1),
        "BBB": _synthetic_ticker("BBB", seed=2),
        "CCC": _synthetic_ticker("CCC", seed=3),
    }
    panel = build_feature_panel(price_map)
    start = panel.index.get_level_values("date").min() + pd.Timedelta(days=400)
    end = panel.index.get_level_values("date").max()
    dates = panel.index.get_level_values("date")
    panel = panel.loc[(dates >= start) & (dates <= end)]
    panel = attach_labels(panel, max_holding_days=5)

    config = StrategyConfig(
        max_holding_days=5,
        train_fraction=0.8,
        entry_min_return=0.0,
        profit_drawdown=0.05,
        max_positions=2,
    )
    wf = fit_predict_walkforward(panel, config)

    assert wf.train_end < wf.test_start
    oos = wf.panel.loc[wf.panel.index.get_level_values("date") >= wf.test_start]
    is_part = wf.panel.loc[wf.panel.index.get_level_values("date") <= wf.train_end]
    assert oos["predicted_return"].notna().any()
    assert not is_part["entry_signal"].fillna(False).any()
    assert "mae" in wf.train_metrics
    assert set(FEATURE_COLUMNS) <= set(wf.feature_importance.index)


def test_run_walkforward_strategy_end_to_end_with_price_map() -> None:
    price_map = {
        "AAA": _synthetic_ticker("AAA", seed=10),
        "BBB": _synthetic_ticker("BBB", seed=11),
        "CCC": _synthetic_ticker("CCC", seed=12),
        "DDD": _synthetic_ticker("DDD", seed=13),
        "EEE": _synthetic_ticker("EEE", seed=14),
    }
    start = price_map["AAA"].index[260].date()
    end = price_map["AAA"].index[-1].date()
    config = StrategyConfig(
        max_positions=3,
        max_holding_days=5,
        train_fraction=0.8,
        entry_min_return=0.0,
        profit_drawdown=0.05,
        cost_bps=0.0,
    )

    result = run_walkforward_strategy(
        list(price_map),
        start=start,
        end=end,
        config=config,
        price_map=price_map,
    )

    assert result.train_end < result.test_start
    assert result.oos_backtest.equity is not None
    assert len(result.oos_backtest.equity) > 0
    assert "total_return" in result.oos_backtest.metrics
    assert result.feature_importance is not None
    assert set(result.tickers) == set(price_map)
