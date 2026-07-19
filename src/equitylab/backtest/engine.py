from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from equitylab.backtest.metrics import compute_metrics


def _at(frame: pd.DataFrame, day: pd.Timestamp, ticker: str) -> float:
    """Typed float read of a single (day, ticker) cell from a wide lookup frame."""
    return float(cast(Any, frame.at[day, ticker]))


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: dict[str, float]
    positions_by_day: pd.Series
    open_positions: pd.DataFrame


@dataclass
class _OpenPosition:
    ticker: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    score: float
    target_hold_days: int
    hold_days: int = 0
    peak_price: float = 0.0


def _empty_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "signal_date",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "shares",
            "pnl",
            "pnl_pct",
            "hold_days",
            "target_hold_days",
            "exit_reason",
            "score",
        ]
    )


def _empty_open_positions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "signal_date",
            "entry_date",
            "asof_date",
            "entry_price",
            "last_price",
            "shares",
            "unrealized_pnl",
            "unrealized_pnl_pct",
            "hold_days",
            "target_hold_days",
            "score",
        ]
    )


def _snapshot_open_positions(
    open_positions: dict[str, _OpenPosition],
    asof: pd.Timestamp,
    close_lookup: pd.DataFrame,
) -> pd.DataFrame:
    if not open_positions:
        return _empty_open_positions()
    rows: list[dict] = []
    for pos in open_positions.values():
        if pos.ticker in close_lookup.columns and not pd.isna(close_lookup.at[asof, pos.ticker]):
            last_price = _at(close_lookup, asof, pos.ticker)
        else:
            last_price = pos.entry_price
        rows.append(
            {
                "ticker": pos.ticker,
                "signal_date": pos.signal_date,
                "entry_date": pos.entry_date,
                "asof_date": asof,
                "entry_price": pos.entry_price,
                "last_price": last_price,
                "shares": pos.shares,
                "unrealized_pnl": pos.shares * (last_price - pos.entry_price),
                "unrealized_pnl_pct": last_price / pos.entry_price - 1.0,
                "hold_days": pos.hold_days,
                "target_hold_days": pos.target_hold_days,
                "score": pos.score,
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def simulate_portfolio(
    panel: pd.DataFrame,
    *,
    max_positions: int = 5,
    max_holding_days: int = 20,
    cost_bps: float = 5.0,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    exit_min_score: float | None = None,
    profit_drawdown: float | None = 0.05,
    model_horizon_exit: bool = False,
    initial_capital: float = 100_000.0,
    signal_col: str = "entry_signal",
    score_col: str = "predicted_return",
    hold_col: str = "predicted_hold_days",
) -> BacktestResult:
    """
    Day loop: fill pending entries, process exits, rank candidates, schedule next-close entries.

    Expects MultiIndex (date, ticker) with close/high/low plus signal/score columns.
    A signal on day t schedules an entry at t+1 close if a slot is free.

    Exits (in order): stop / take-profit / profit_drawdown / model score drop /
    model_horizon_exit (hold_days reaches predicted best day) / max_holding_days.

    Positions still open at the last bar are left open (no synthetic end_of_data
    exit) and returned in ``open_positions`` with model score / target hold.
    """
    if max_positions < 1:
        raise ValueError("max_positions must be >= 1")
    if max_holding_days < 1:
        raise ValueError("max_holding_days must be >= 1")
    if panel.empty:
        empty_eq = pd.Series(dtype=float, name="equity")
        empty_trades = _empty_trades()
        return BacktestResult(
            equity=empty_eq,
            trades=empty_trades,
            metrics=compute_metrics(empty_eq, empty_trades),
            positions_by_day=pd.Series(dtype=float, name="n_positions"),
            open_positions=_empty_open_positions(),
        )

    required = {"close", "high", "low", signal_col, score_col}
    if model_horizon_exit:
        required.add(hold_col)
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")

    dates = panel.index.get_level_values("date").unique().sort_values()
    cost = cost_bps / 10_000.0
    cash = float(initial_capital)
    open_positions: dict[str, _OpenPosition] = {}
    # (entry_date, signal_date, ticker, score, target_hold_days)
    pending: list[tuple[pd.Timestamp, pd.Timestamp, str, float, int]] = []
    trades: list[dict] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []
    position_counts: list[tuple[pd.Timestamp, int]] = []
    fully_invested_days = 0

    close_lookup = panel["close"].unstack("ticker")
    high_lookup = panel["high"].unstack("ticker")
    low_lookup = panel["low"].unstack("ticker")
    score_lookup = panel[score_col].unstack("ticker")

    def mark_to_market(asof: pd.Timestamp) -> float:
        value = cash
        for pos in open_positions.values():
            if pos.ticker in close_lookup.columns and not pd.isna(close_lookup.at[asof, pos.ticker]):
                price = _at(close_lookup, asof, pos.ticker)
            else:
                price = pos.entry_price
            value += pos.shares * price
        return value

    def close_position(ticker: str, day: pd.Timestamp, exit_raw: float, reason: str) -> None:
        nonlocal cash
        pos = open_positions.pop(ticker)
        exit_price = float(exit_raw) * (1.0 - cost)
        proceeds = pos.shares * exit_price
        cash += proceeds
        trades.append(
            {
                "ticker": ticker,
                "signal_date": pos.signal_date,
                "entry_date": pos.entry_date,
                "exit_date": day,
                "entry_price": pos.entry_price,
                "exit_price": exit_price,
                "shares": pos.shares,
                "pnl": proceeds - pos.shares * pos.entry_price,
                "pnl_pct": exit_price / pos.entry_price - 1.0,
                "hold_days": pos.hold_days,
                "target_hold_days": pos.target_hold_days,
                "exit_reason": reason,
                "score": pos.score,
            }
        )

    for i, day in enumerate(dates):
        # Fill entries scheduled for today.
        still_pending: list[tuple[pd.Timestamp, pd.Timestamp, str, float, int]] = []
        for entry_date, signal_date, ticker, score, target_hold in pending:
            if entry_date != day:
                still_pending.append((entry_date, signal_date, ticker, score, target_hold))
                continue
            if ticker in open_positions or len(open_positions) >= max_positions:
                continue
            if ticker not in close_lookup.columns or pd.isna(close_lookup.at[day, ticker]):
                continue

            equity_now = mark_to_market(day)
            alloc = equity_now / max_positions
            raw_price = _at(close_lookup, day, ticker)
            entry_price = raw_price * (1.0 + cost)
            if entry_price <= 0 or alloc <= 0:
                continue
            shares = alloc / entry_price
            cash -= shares * entry_price
            open_positions[ticker] = _OpenPosition(
                ticker=ticker,
                signal_date=signal_date,
                entry_date=day,
                entry_price=entry_price,
                shares=shares,
                score=score,
                target_hold_days=target_hold,
                hold_days=0,
                peak_price=raw_price,
            )
        pending = still_pending

        # Age and exit positions (skip same-day entries).
        for ticker, pos in list(open_positions.items()):
            if pos.entry_date == day:
                continue
            if ticker not in close_lookup.columns or pd.isna(close_lookup.at[day, ticker]):
                continue

            pos.hold_days += 1
            high = _at(high_lookup, day, ticker)
            low = _at(low_lookup, day, ticker)
            close = _at(close_lookup, day, ticker)
            pos.peak_price = max(pos.peak_price, high)

            if stop_loss is not None and low <= pos.entry_price * (1.0 + stop_loss):
                close_position(ticker, day, pos.entry_price * (1.0 + stop_loss), "stop_loss")
                continue
            if take_profit is not None and high >= pos.entry_price * (1.0 + take_profit):
                close_position(ticker, day, pos.entry_price * (1.0 + take_profit), "take_profit")
                continue
            if profit_drawdown is not None and pos.peak_price > 0:
                trail_stop = pos.peak_price * (1.0 - profit_drawdown)
                if close <= trail_stop:
                    close_position(ticker, day, close, "profit_drawdown")
                    continue
            if exit_min_score is not None and ticker in score_lookup.columns:
                day_score = score_lookup.at[day, ticker]
                if pd.notna(day_score) and float(cast(Any, day_score)) < exit_min_score:
                    close_position(ticker, day, close, "model_exit")
                    continue
            if model_horizon_exit and pos.hold_days >= pos.target_hold_days:
                close_position(ticker, day, close, "model_horizon")
                continue
            if pos.hold_days >= max_holding_days:
                close_position(ticker, day, close, "max_hold")

        # Schedule new entries for next close.
        if i + 1 < len(dates):
            next_day = dates[i + 1]
            pending_next = sum(1 for ed, _, _, _, _ in pending if ed == next_day)
            free = max_positions - len(open_positions) - pending_next
            if free > 0:
                try:
                    day_slice = cast(pd.DataFrame, panel.xs(day, level="date"))
                except KeyError:
                    day_slice = pd.DataFrame()
                if not day_slice.empty:
                    candidates = day_slice.loc[day_slice[signal_col].fillna(False)]
                    blocked = set(open_positions) | {t for _, _, t, _, _ in pending}
                    candidates = candidates[~candidates.index.isin(blocked)]
                    if not candidates.empty:
                        ranked = candidates.sort_values(by=score_col, ascending=False)
                        for cand_ticker, row in ranked.head(free).iterrows():
                            if model_horizon_exit and hold_col in row.index and pd.notna(row[hold_col]):
                                target_hold = int(max(1, min(max_holding_days, round(float(row[hold_col])))))
                            else:
                                target_hold = max_holding_days
                            pending.append(
                                (next_day, day, str(cand_ticker), float(row[score_col]), target_hold)
                            )

        n_open = len(open_positions)
        position_counts.append((day, n_open))
        if n_open >= max_positions:
            fully_invested_days += 1
        equity_points.append((day, mark_to_market(day)))

    last = dates[-1] if dates.size else None
    open_df = (
        _snapshot_open_positions(open_positions, last, close_lookup)
        if last is not None
        else _empty_open_positions()
    )

    equity = pd.Series({d: v for d, v in equity_points}, name="equity", dtype=float).sort_index()
    trades_df = pd.DataFrame(trades) if trades else _empty_trades()
    positions_by_day = pd.Series(
        {d: n for d, n in position_counts},
        name="n_positions",
        dtype=float,
    ).sort_index()

    metrics = compute_metrics(equity, trades_df)
    metrics["pct_days_fully_invested"] = (
        fully_invested_days / len(dates) if len(dates) else 0.0
    )
    metrics["open_position_count"] = float(len(open_df))

    return BacktestResult(
        equity=equity,
        trades=trades_df,
        metrics=metrics,
        positions_by_day=positions_by_day,
        open_positions=open_df,
    )
