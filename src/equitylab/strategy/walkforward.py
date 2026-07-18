from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from equitylab.signals.features import FEATURE_COLUMNS
from equitylab.signals.labels import horizon_columns
from equitylab.strategy.config import StrategyConfig

SCORE_COL = "predicted_return"
HOLD_COL = "predicted_hold_days"


@dataclass(frozen=True)
class WalkForwardModelResult:
    model: MultiOutputRegressor
    feature_columns: list[str]
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importance: pd.Series
    panel: pd.DataFrame


def chronological_split_dates(
    dates: pd.DatetimeIndex,
    train_fraction: float,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (last_train_date, first_test_date) for a chronological split."""
    unique = pd.DatetimeIndex(sorted(dates.unique()))
    if len(unique) < 2:
        raise ValueError("Need at least 2 distinct dates for walk-forward split")
    split_idx = int(len(unique) * train_fraction)
    split_idx = min(max(split_idx, 1), len(unique) - 1)
    train_end = unique[split_idx - 1]
    test_start = unique[split_idx]
    return train_end, test_start


def label_embargo_cutoff(
    dates: pd.DatetimeIndex,
    train_end: pd.Timestamp,
    max_holding_days: int,
) -> pd.Timestamp:
    """
    Last feature date whose longest forward label ends on or before train_end.

    Labels use closes through t+N; require t+N <= train_end in trading-day space.
    """
    if max_holding_days < 1:
        raise ValueError("max_holding_days must be >= 1")

    unique = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    train_dates = unique[unique <= train_end]
    if len(train_dates) <= max_holding_days:
        raise ValueError(
            f"Not enough train dates ({len(train_dates)}) for "
            f"max_holding_days={max_holding_days} embargo"
        )
    return pd.Timestamp(train_dates[-(max_holding_days + 1)])


def fit_predict_walkforward(
    panel: pd.DataFrame,
    config: StrategyConfig,
    *,
    random_state: int = 42,
) -> WalkForwardModelResult:
    """
    Train a multi-horizon regressor on forward returns for every day 1..N.

    y = [close[t+1]/close[t]-1, ..., close[t+N]/close[t]-1]
    Score for ranking/entry = max predicted horizon return (best expected exit day).

    Train rows are embargoed so labels cannot reach past train_end into the test window.
    """
    if panel.empty:
        raise ValueError("panel is empty")

    missing = [c for c in FEATURE_COLUMNS if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing feature columns: {missing}")

    y_cols = horizon_columns(config.max_holding_days)
    missing_y = [c for c in y_cols if c not in panel.columns]
    if missing_y:
        raise ValueError(f"panel missing horizon label columns: {missing_y}")

    dates = panel.index.get_level_values("date")
    train_end, test_start = chronological_split_dates(dates, config.train_fraction)
    train_label_end = label_embargo_cutoff(
        dates, train_end, config.max_holding_days
    )

    train_mask = dates <= train_label_end
    test_mask = dates >= test_start

    train = panel.loc[train_mask].dropna(subset=[*FEATURE_COLUMNS, *y_cols])
    test = panel.loc[test_mask].dropna(subset=FEATURE_COLUMNS)

    if train.empty:
        raise ValueError("No valid training rows after embargo / dropping NaNs")
    if test.empty:
        raise ValueError("No valid test rows after dropping NaNs")

    y_train = train[y_cols].astype(float)
    if float(y_train.stack().std()) == 0.0:
        raise ValueError("Training forward returns have zero variance")

    base = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.1,
        max_iter=100,
        random_state=random_state,
    )
    model = MultiOutputRegressor(base)
    x_train = train[FEATURE_COLUMNS]
    model.fit(x_train, y_train)

    def _predict_horizons(frame: pd.DataFrame) -> np.ndarray:
        """Predicted returns shape (n_rows, N) for horizons 1..N."""
        return np.asarray(model.predict(frame[FEATURE_COLUMNS]), dtype=float)

    train_horizons = _predict_horizons(train)
    test_horizons = _predict_horizons(test)
    train_pred_ret = train_horizons.max(axis=1)
    test_pred_ret = test_horizons.max(axis=1)
    # 1-based hold day with the best predicted return
    train_pred_hold = train_horizons.argmax(axis=1) + 1
    test_pred_hold = test_horizons.argmax(axis=1) + 1

    train_signal = train_pred_ret >= config.entry_min_return
    test_signal = test_pred_ret >= config.entry_min_return

    # Metrics vs best realized horizon return (same aggregation as the score).
    y_train_best = y_train.max(axis=1)
    train_metrics = {
        "mae": float(mean_absolute_error(y_train_best, train_pred_ret)),
        "r2": float(r2_score(y_train_best, train_pred_ret)),
        "n_rows": float(len(train)),
        "mean_forward_return": float(y_train_best.mean()),
        "signal_rate": float(train_signal.mean()),
        "embargo_days": float(config.max_holding_days),
        "n_horizons": float(len(y_cols)),
        "mean_predicted_hold_days": float(np.mean(train_pred_hold)),
        "train_label_end": float(pd.Timestamp(train_label_end).value),
    }

    test_metrics = {
        "n_rows": float(len(test)),
        "signal_rate": float(test_signal.mean()),
        "mean_predicted_return": float(np.mean(test_pred_ret)),
        "mean_predicted_hold_days": float(np.mean(test_pred_hold)),
        "n_horizons": float(len(y_cols)),
    }
    test_labeled = test.dropna(subset=y_cols)
    if len(test_labeled) > 1:
        y_test_best = test_labeled[y_cols].astype(float).max(axis=1)
        pred_test = _predict_horizons(test_labeled).max(axis=1)
        test_metrics["mae"] = float(mean_absolute_error(y_test_best, pred_test))
        test_metrics["r2"] = float(r2_score(y_test_best, pred_test))
        test_metrics["mean_forward_return"] = float(y_test_best.mean())
    else:
        test_metrics["mae"] = float("nan")
        test_metrics["r2"] = float("nan")
        test_metrics["mean_forward_return"] = float("nan")

    scored = panel.copy()
    scored[SCORE_COL] = np.nan
    scored[HOLD_COL] = np.nan
    scored["entry_signal"] = False
    scored["entry_signal_is"] = False
    scored.loc[train.index, SCORE_COL] = train_pred_ret
    scored.loc[test.index, SCORE_COL] = test_pred_ret
    scored.loc[train.index, HOLD_COL] = train_pred_hold.astype(float)
    scored.loc[test.index, HOLD_COL] = test_pred_hold.astype(float)
    scored.loc[test.index, "entry_signal"] = test_signal
    scored.loc[train.index, "entry_signal_is"] = train_signal

    sample = train
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=random_state)
    try:
        # Importance from the first horizon estimator (representative).
        first_est = model.estimators_[0]
        imp = permutation_importance(
            first_est,
            sample[FEATURE_COLUMNS],
            sample[y_cols[0]].astype(float),
            n_repeats=5,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        feature_importance = pd.Series(
            imp.importances_mean,
            index=FEATURE_COLUMNS,
            name="importance",
        ).sort_values(ascending=False)
    except Exception:  # noqa: BLE001
        feature_importance = pd.Series(0.0, index=FEATURE_COLUMNS, name="importance")

    return WalkForwardModelResult(
        model=model,
        feature_columns=list(FEATURE_COLUMNS),
        train_end=pd.Timestamp(train_end),
        test_start=pd.Timestamp(test_start),
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        feature_importance=feature_importance,
        panel=scored,
    )
