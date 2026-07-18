# EquityLab

A small research platform for screening US equities and evaluating a machine-learning
trading strategy with a **leakage-safe, expanding walk-forward** backtest — end to end,
from a live Yahoo Finance universe query to a portfolio-level P&L, exposed through a
Streamlit UI.

```
Yahoo universe query → technical screen → feature/label panel
    → walk-forward ML (train/predict per fold, embargoed) → portfolio backtest
```

## Why walk-forward, not a single train/test split

Fitting one model on the first N% of history and testing on the rest overstates how a
strategy would actually perform, because market regimes drift and a single split hides
that. This project instead:

1. Trains on an initial window, predicts the next out-of-sample chunk (`test_step_days`,
   default ~1 month), then **retrains on all history so far** and repeats — an expanding
   walk-forward, so every prediction is made by a model that has only ever seen the past.
2. **Embargoes labels at the fold boundary.** Labels are forward returns (`close[t+k]/close[t]-1`
   for k = 1..`max_holding_days`), so a label computed near `train_end` can reach past it.
   `label_embargo_cutoff` drops the last `max_holding_days` days of the training window so no
   training label ever looks past the point the model is evaluated at
   ([`walkforward.py`](src/equitylab/strategy/walkforward.py)).
3. Glues the out-of-sample predictions from every fold together and runs a single portfolio
   backtest over that OOS stream — so the reported Sharpe/return/drawdown reflect only
   predictions the model made without having seen the answer.

The model predicts a forward return for every hold horizon 1..N and takes the max
(best exit day within the window) as its score; the portfolio backtest enters at the next
day's close, ranks candidates by predicted return when slots are free, and exits on
whichever of stop-loss / take-profit / trailing profit-drawdown / model-score decay /
predicted-horizon / max-holding-days fires first. Positions still open at the end of the
data are reported as open, not force-closed — closing them artificially would distort the
realized P&L.

## Project layout

```
src/equitylab/
  data/           Yahoo Finance loader + a DuckDB-backed local price cache
                  (only missing date ranges hit the network)
  screening/      Yahoo EquityQuery universe fetch (pre_yahoo) + technical
                  filters/scoring (post_yahoo): 52w drawdown, RSI, relative
                  volume, distance from SMA200
  signals/        Feature engineering (RSI, ATR%, MACD histogram, realized
                  vol, EWMA distance, ...) and per-horizon forward-return labels
  strategy/       StrategyConfig, the walk-forward fit/predict loop, and the
                  pipeline that wires loaders → features → walk-forward → backtest
  backtest/       Portfolio simulator (next-close fills, position sizing,
                  multiple exit rules, transaction costs) + performance metrics
src/main.py       Streamlit app: configure a screen, run it, then configure
                  and run the walk-forward strategy over the results
```

## Running it

Requires Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run streamlit run src/main.py
```

## Tests

```bash
uv run pytest
```

43 tests cover fold construction, the label-embargo boundary, walk-forward fit/predict
across every supported model, and portfolio-engine mechanics (entry timing, exit
precedence, position sizing, open-position handling) against synthetic price series with
exact expected trades — not just "does it run" smoke tests.

## Models

Selectable per run: `HistGradientBoostingRegressor`, `XGBoost`, `RandomForestRegressor`,
`Ridge` — each wrapped in a `MultiOutputRegressor` to predict all `1..max_holding_days`
forward-return horizons at once ([`strategy/walkforward.py`](src/equitylab/strategy/walkforward.py)).

## Disclaimer

Research/educational tooling, not investment advice. Yahoo Finance data via `yfinance`
may be delayed, incomplete, or revised.
