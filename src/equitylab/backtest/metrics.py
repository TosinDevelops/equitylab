from __future__ import annotations

import math

import pandas as pd


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    *,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """Portfolio performance metrics from equity curve and trade blotter."""
    if equity.empty:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "avg_hold_days": 0.0,
            "trade_count": 0.0,
            "pct_days_fully_invested": 0.0,
        }

    equity = equity.astype(float).sort_index()
    start_value = float(equity.iloc[0])
    end_value = float(equity.iloc[-1])
    total_return = end_value / start_value - 1.0 if start_value else 0.0

    n_days = max(len(equity) - 1, 0)
    years = n_days / trading_days_per_year if n_days else 0.0
    if years > 0 and start_value > 0 and end_value > 0:
        cagr = (end_value / start_value) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    daily_ret = equity.pct_change().dropna()
    if len(daily_ret) > 1 and float(daily_ret.std()) > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(trading_days_per_year))
    else:
        sharpe = 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    if trades is not None and not trades.empty and "pnl_pct" in trades.columns:
        wins = trades["pnl_pct"] > 0
        win_rate = float(wins.mean())
        avg_hold = float(trades["hold_days"].mean()) if "hold_days" in trades.columns else 0.0
        trade_count = float(len(trades))
    else:
        win_rate = 0.0
        avg_hold = 0.0
        trade_count = 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": win_rate,
        "avg_hold_days": avg_hold,
        "trade_count": trade_count,
        "pct_days_fully_invested": float("nan"),  # filled by engine when available
    }
