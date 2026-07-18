from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    # portfolio / execution
    max_positions: int = 5
    max_holding_days: int = 20
    cost_bps: float = 5.0
    stop_loss: float | None = None
    take_profit: float | None = None
    # Exit when close falls this far below the peak close since entry (e.g. 0.05 = 5%).
    profit_drawdown: float | None = 0.05
    # If True, exit at the model's predicted best hold day (argmax over horizons 1..N).
    model_horizon_exit: bool = False
    # walk-forward / ML: model predicts buy→sell profit over max_holding_days
    train_fraction: float = 0.80
    # Enter when predicted N-day return >= this.
    entry_min_return: float = 0.0
    # Optional model exit: when predicted return falls below this (None = disabled).
    exit_min_return: float | None = None
    initial_capital: float = 100_000.0

    def __post_init__(self) -> None:
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1 (exclusive)")
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        if self.max_holding_days < 1:
            raise ValueError("max_holding_days must be >= 1")
        if self.profit_drawdown is not None and not (0.0 < self.profit_drawdown < 1.0):
            raise ValueError("profit_drawdown must be between 0 and 1 (exclusive)")
        if (
            self.exit_min_return is not None
            and self.exit_min_return > self.entry_min_return
        ):
            raise ValueError("exit_min_return should be <= entry_min_return (hysteresis)")
