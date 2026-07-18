from __future__ import annotations

import pandas as pd
import pytest

from equitylab.backtest.engine import simulate_portfolio


def _panel_from_closes(
    closes: dict[str, list[float]],
    signals: dict[str, list[bool]],
    scores: dict[str, list[float]],
    holds: dict[str, list[float]] | None = None,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    n = len(next(iter(closes.values())))
    dates = pd.date_range(start, periods=n, freq="B")
    rows = []
    for ticker, series in closes.items():
        for i, day in enumerate(dates):
            price = series[i]
            row = {
                "date": day,
                "ticker": ticker,
                "open": price,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "entry_signal": signals[ticker][i],
                "predicted_return": scores[ticker][i],
            }
            if holds is not None:
                row["predicted_hold_days"] = holds[ticker][i]
            rows.append(row)
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def test_entry_at_next_close_and_exit_at_max_hold() -> None:
    # Signal on day 0 for AAA → enter day 1 close=101, hold 2 days → exit day 3 close=103
    closes = {"AAA": [100.0, 101.0, 102.0, 103.0, 104.0]}
    signals = {"AAA": [True, False, False, False, False]}
    scores = {"AAA": [0.05, 0.05, 0.05, 0.05, 0.05]}
    panel = _panel_from_closes(closes, signals, scores)

    result = simulate_portfolio(
        panel,
        max_positions=1,
        max_holding_days=2,
        cost_bps=0.0,
        exit_min_score=None,  # force max-hold path
        initial_capital=10_000.0,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["ticker"] == "AAA"
    assert trade["entry_date"] == panel.index.get_level_values("date").unique()[1]
    assert trade["entry_price"] == pytest.approx(101.0)
    assert trade["hold_days"] == 2
    assert trade["exit_reason"] == "max_hold"
    assert trade["exit_price"] == pytest.approx(103.0)


def test_full_slots_skip_lower_score_candidates() -> None:
    # Day 0: three signals, max_positions=1 → only highest predicted return enters
    closes = {
        "AAA": [100.0, 100.0, 100.0, 100.0],
        "BBB": [100.0, 100.0, 100.0, 100.0],
        "CCC": [100.0, 100.0, 100.0, 100.0],
    }
    signals = {
        "AAA": [True, False, False, False],
        "BBB": [True, False, False, False],
        "CCC": [True, False, False, False],
    }
    scores = {
        "AAA": [0.02, 0.0, 0.0, 0.0],
        "BBB": [0.08, 0.0, 0.0, 0.0],
        "CCC": [0.04, 0.0, 0.0, 0.0],
    }
    panel = _panel_from_closes(closes, signals, scores)

    result = simulate_portfolio(
        panel,
        max_positions=1,
        max_holding_days=2,
        cost_bps=0.0,
        exit_min_score=None,
    )

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["ticker"] == "BBB"
    assert result.trades.iloc[0]["score"] == pytest.approx(0.08)


def test_profit_drawdown_exits_from_peak() -> None:
    # Enter 100; peak high rises to ~112 (close 110 * 1.02); then close 100 → >5% off peak
    closes = {"AAA": [100.0, 100.0, 110.0, 100.0, 100.0]}
    signals = {"AAA": [True, False, False, False, False]}
    scores = {"AAA": [0.05, 0.05, 0.05, 0.05, 0.05]}
    panel = _panel_from_closes(closes, signals, scores)

    result = simulate_portfolio(
        panel,
        max_positions=1,
        max_holding_days=10,
        cost_bps=0.0,
        exit_min_score=None,
        profit_drawdown=0.05,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "profit_drawdown"
    assert trade["exit_price"] == pytest.approx(100.0)


def test_model_horizon_exit_on_predicted_hold_day() -> None:
    # Signal day 0 with predicted hold = 2 → enter day 1, exit when hold_days hits 2
    closes = {"AAA": [100.0, 100.0, 101.0, 102.0, 103.0]}
    signals = {"AAA": [True, False, False, False, False]}
    scores = {"AAA": [0.05, 0.05, 0.05, 0.05, 0.05]}
    holds = {"AAA": [2.0, 2.0, 2.0, 2.0, 2.0]}
    panel = _panel_from_closes(closes, signals, scores, holds=holds)

    result = simulate_portfolio(
        panel,
        max_positions=1,
        max_holding_days=10,
        cost_bps=0.0,
        exit_min_score=None,
        profit_drawdown=None,
        model_horizon_exit=True,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "model_horizon"
    assert trade["hold_days"] == 2
    assert trade["target_hold_days"] == 2
    assert trade["exit_price"] == pytest.approx(102.0)


def test_model_exit_when_predicted_return_drops() -> None:
    closes = {"AAA": [100.0, 100.0, 101.0, 102.0, 103.0]}
    signals = {"AAA": [True, False, False, False, False]}
    scores = {"AAA": [0.05, 0.05, -0.03, -0.03, -0.03]}
    panel = _panel_from_closes(closes, signals, scores)

    result = simulate_portfolio(
        panel,
        max_positions=1,
        max_holding_days=10,
        cost_bps=0.0,
        exit_min_score=-0.01,
        profit_drawdown=None,
    )

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "model_exit"
    assert trade["hold_days"] == 1
    assert trade["exit_price"] == pytest.approx(101.0)


def test_open_positions_left_open_at_end_of_data() -> None:
    # Enter day 1, max_hold=10, only 4 bars → still open at the end (no end_of_data exit).
    closes = {"AAA": [100.0, 101.0, 102.0, 103.0]}
    signals = {"AAA": [True, False, False, False]}
    scores = {"AAA": [0.07, 0.07, 0.07, 0.07]}
    holds = {"AAA": [5.0, 5.0, 5.0, 5.0]}
    panel = _panel_from_closes(closes, signals, scores, holds=holds)

    result = simulate_portfolio(
        panel,
        max_positions=1,
        max_holding_days=10,
        cost_bps=0.0,
        exit_min_score=None,
        profit_drawdown=None,
        model_horizon_exit=True,
        initial_capital=10_000.0,
    )

    assert result.trades.empty
    assert len(result.open_positions) == 1
    open_row = result.open_positions.iloc[0]
    assert open_row["ticker"] == "AAA"
    assert open_row["score"] == pytest.approx(0.07)
    assert open_row["target_hold_days"] == 5
    assert open_row["hold_days"] == 2  # aged on days 2 and 3
    assert open_row["last_price"] == pytest.approx(103.0)
    assert result.metrics["open_position_count"] == 1.0
    # Equity still marks open stock to market (not forced to cash).
    shares = 10_000.0 / 101.0
    assert result.equity.iloc[-1] == pytest.approx(shares * 103.0)


def test_stop_loss_exits_early() -> None:
    closes = {"AAA": [100.0, 100.0, 90.0, 90.0]}
    signals = {"AAA": [True, False, False, False]}
    scores = {"AAA": [0.05, 0.05, 0.05, 0.05]}
    panel = _panel_from_closes(closes, signals, scores)
    day = panel.index.get_level_values("date").unique()[2]
    panel.loc[(day, "AAA"), "low"] = 90.0

    result = simulate_portfolio(
        panel,
        max_positions=1,
        max_holding_days=10,
        cost_bps=0.0,
        stop_loss=-0.05,
        exit_min_score=None,
    )
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "stop_loss"
