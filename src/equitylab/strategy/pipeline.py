from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import pandas as pd

from equitylab.backtest.engine import BacktestResult, simulate_portfolio
from equitylab.data.loaders.yahoo import load_data, normalize_ticker
from equitylab.signals.features import build_feature_panel
from equitylab.signals.labels import attach_labels
from equitylab.strategy.config import StrategyConfig
from equitylab.strategy.walkforward import (
    WalkForwardFoldResult,
    WalkForwardModelResult,
    fit_predict_walkforward,
)


@dataclass(frozen=True)
class StrategyResult:
    config: StrategyConfig
    tickers: list[str]
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importance: pd.Series
    oos_backtest: BacktestResult
    is_backtest: BacktestResult
    panel: pd.DataFrame
    errors: list[str]
    folds: list[WalkForwardFoldResult]


def _load_price_map(
    tickers: list[str],
    start: date,
    end: date,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    lookback_start = start - timedelta(days=450)
    price_map: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    total = max(len(tickers), 1)

    for index, raw in enumerate(tickers):
        ticker = normalize_ticker(raw)
        if not ticker:
            continue
        if progress is not None:
            progress((index + 1) / total, f"Loading {ticker}")
        try:
            prices = load_data(
                ticker,
                interval="1d",
                start=lookback_start.isoformat(),
                end=end.isoformat(),
            )
            if len(prices) < 220:
                errors.append(f"{ticker}: not enough history ({len(prices)} bars)")
                continue
            price_map[ticker] = prices
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ticker}: {exc}")

    return price_map, errors


def run_walkforward_strategy(
    tickers: list[str],
    start: date,
    end: date,
    config: StrategyConfig | None = None,
    *,
    price_map: dict[str, pd.DataFrame] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> StrategyResult:
    """
    Load prices → features/labels → expanding walk-forward → OOS portfolio backtest.

    Pass price_map in tests to avoid network I/O.
    """
    config = config or StrategyConfig()
    errors: list[str] = []

    if price_map is None:
        price_map, errors = _load_price_map(tickers, start, end, progress=progress)
    else:
        price_map = {
            normalize_ticker(t): df for t, df in price_map.items() if normalize_ticker(t)
        }

    if not price_map:
        raise ValueError(errors[0] if errors else "No price data available")

    if progress is not None:
        progress(0.85, "Building features and labels…")

    panel = build_feature_panel(price_map)
    # Restrict strategy window to [start, end] after features used full lookback.
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    dates = panel.index.get_level_values("date")
    panel = panel.loc[(dates >= start_ts) & (dates <= end_ts)]
    panel = attach_labels(panel, max_holding_days=config.max_holding_days)

    if panel.empty:
        raise ValueError("Feature panel is empty for the selected date range")

    if progress is not None:
        progress(0.9, "Fitting walk-forward folds…")

    wf: WalkForwardModelResult = fit_predict_walkforward(
        panel, config, progress=progress
    )
    scored = wf.panel

    oos = scored.loc[scored.index.get_level_values("date") >= wf.test_start]
    is_panel = scored.loc[scored.index.get_level_values("date") <= wf.train_end].copy()
    if "entry_signal_is" in is_panel.columns:
        is_panel["entry_signal"] = is_panel["entry_signal_is"].fillna(False)

    if progress is not None:
        progress(0.95, "Running portfolio backtests…")

    oos_bt = simulate_portfolio(
        oos,
        max_positions=config.max_positions,
        max_holding_days=config.max_holding_days,
        cost_bps=config.cost_bps,
        stop_loss=config.stop_loss,
        take_profit=config.take_profit,
        exit_min_score=config.exit_min_return,
        profit_drawdown=config.profit_drawdown,
        model_horizon_exit=config.model_horizon_exit,
        initial_capital=config.initial_capital,
        score_col="predicted_return",
        hold_col="predicted_hold_days",
    )
    is_bt = simulate_portfolio(
        is_panel,
        max_positions=config.max_positions,
        max_holding_days=config.max_holding_days,
        cost_bps=config.cost_bps,
        stop_loss=config.stop_loss,
        take_profit=config.take_profit,
        exit_min_score=config.exit_min_return,
        profit_drawdown=config.profit_drawdown,
        model_horizon_exit=config.model_horizon_exit,
        initial_capital=config.initial_capital,
        score_col="predicted_return",
        hold_col="predicted_hold_days",
    )

    if progress is not None:
        progress(1.0, "Done")

    return StrategyResult(
        config=config,
        tickers=sorted(price_map),
        train_end=wf.train_end,
        test_start=wf.test_start,
        train_metrics=wf.train_metrics,
        test_metrics=wf.test_metrics,
        feature_importance=wf.feature_importance,
        oos_backtest=oos_bt,
        is_backtest=is_bt,
        panel=scored,
        errors=errors,
        folds=list(wf.folds),
    )
