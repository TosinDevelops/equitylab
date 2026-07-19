from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from equitylab.signals.features import FEATURE_COLUMNS
from equitylab.signals.labels import horizon_columns
from equitylab.strategy.config import ModelName, StrategyConfig

SCORE_COL = "predicted_return"
HOLD_COL = "predicted_hold_days"


def make_base_estimator(model_name: ModelName, random_state: int = 42):
    """Build the sklearn-compatible base regressor for one forward-return horizon."""
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.1,
            max_iter=100,
            random_state=random_state,
        )
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "ridge":
        return Ridge(alpha=1.0)
    raise ValueError(f"Unknown model_name: {model_name!r}")


@dataclass(frozen=True)
class WalkForwardFoldResult:
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]


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
    folds: list[WalkForwardFoldResult]


def chronological_split_dates(
    dates: pd.Index,
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


def iter_walkforward_folds(
    dates: pd.Index,
    train_fraction: float,
    test_step_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Expanding walk-forward folds: (train_end, test_start, test_end).

    First train ends at train_fraction; each OOS chunk is test_step_days long
    (last chunk may be shorter). Train expands up to the day before each test.
    """
    if test_step_days < 1:
        raise ValueError("test_step_days must be >= 1")

    unique = pd.DatetimeIndex(sorted(pd.DatetimeIndex(dates).unique()))
    _, first_test_start = chronological_split_dates(unique, train_fraction)
    test_start_idx = cast(int, unique.get_loc(first_test_start))

    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    while test_start_idx < len(unique):
        test_end_idx = min(test_start_idx + test_step_days - 1, len(unique) - 1)
        train_end = pd.Timestamp(unique[test_start_idx - 1])
        test_start = pd.Timestamp(unique[test_start_idx])
        test_end = pd.Timestamp(unique[test_end_idx])
        folds.append((train_end, test_start, test_end))
        test_start_idx = test_end_idx + 1
    return folds


def label_embargo_cutoff(
    dates: pd.Index,
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


def _fold_metrics(
    y_best: pd.Series,
    pred_ret: np.ndarray,
    pred_hold: np.ndarray,
    signal: np.ndarray,
    *,
    n_horizons: int,
    embargo_days: int,
    train_label_end: pd.Timestamp | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "n_rows": float(len(y_best)),
        "signal_rate": float(np.mean(signal)) if len(signal) else float("nan"),
        "mean_predicted_return": float(np.mean(pred_ret)) if len(pred_ret) else float("nan"),
        "mean_predicted_hold_days": float(np.mean(pred_hold)) if len(pred_hold) else float("nan"),
        "n_horizons": float(n_horizons),
        "embargo_days": float(embargo_days),
    }
    if len(y_best) > 1:
        metrics["mae"] = float(mean_absolute_error(y_best, pred_ret))
        metrics["r2"] = float(r2_score(y_best, pred_ret))
        metrics["mean_forward_return"] = float(y_best.mean())
    else:
        metrics["mae"] = float("nan")
        metrics["r2"] = float("nan")
        metrics["mean_forward_return"] = float("nan")
    if train_label_end is not None:
        metrics["train_label_end"] = float(pd.Timestamp(train_label_end).value)
    return metrics


def _permutation_importance(
    model: MultiOutputRegressor,
    sample: pd.DataFrame,
    y_cols: list[str],
    *,
    random_state: int,
) -> pd.Series:
    if sample.empty:
        return pd.Series(0.0, index=FEATURE_COLUMNS, name="importance")
    if len(sample) > 2000:
        sample = sample.sample(2000, random_state=random_state)
    try:
        first_est = model.estimators_[0]
        imp = permutation_importance(
            first_est,
            sample[FEATURE_COLUMNS],
            sample[y_cols[0]].astype(float),
            n_repeats=5,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        return pd.Series(
            imp.importances_mean,
            index=FEATURE_COLUMNS,
            name="importance",
        ).sort_values(ascending=False)
    except Exception:  # noqa: BLE001
        return pd.Series(0.0, index=FEATURE_COLUMNS, name="importance")


def fit_predict_walkforward(
    panel: pd.DataFrame,
    config: StrategyConfig,
    *,
    random_state: int = 42,
    progress: Callable[[float, str], None] | None = None,
) -> WalkForwardModelResult:
    """
    Expanding walk-forward: retrain each fold, score only that fold's OOS chunk.

    y = [close[t+1]/close[t]-1, ..., close[t+N]/close[t]-1]
    Score for ranking/entry = max predicted horizon return (best expected exit day).

    Train rows are embargoed so labels cannot reach past each fold's train_end.
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
    fold_windows = iter_walkforward_folds(
        dates, config.train_fraction, config.test_step_days
    )
    if not fold_windows:
        raise ValueError("No walk-forward folds produced")

    scored = panel.copy()
    scored[SCORE_COL] = np.nan
    scored[HOLD_COL] = np.nan
    scored["entry_signal"] = False
    scored["entry_signal_is"] = False

    fold_results: list[WalkForwardFoldResult] = []
    importance_frames: list[pd.Series] = []
    oos_y_parts: list[pd.Series] = []
    oos_pred_parts: list[np.ndarray] = []
    oos_signal_parts: list[np.ndarray] = []
    oos_hold_parts: list[np.ndarray] = []
    first_train_metrics: dict[str, float] | None = None
    last_model: MultiOutputRegressor | None = None
    n_folds = len(fold_windows)

    for fold_i, (train_end, test_start, test_end) in enumerate(fold_windows):
        if progress is not None:
            progress(
                0.9 + 0.05 * (fold_i / n_folds),
                f"Fitting walk-forward fold {fold_i + 1}/{n_folds}…",
            )

        train_label_end = label_embargo_cutoff(
            dates, train_end, config.max_holding_days
        )
        train_mask = dates <= train_label_end
        test_mask = (dates >= test_start) & (dates <= test_end)

        train = panel.loc[train_mask].dropna(subset=[*FEATURE_COLUMNS, *y_cols])
        test = panel.loc[test_mask].dropna(subset=FEATURE_COLUMNS)

        if train.empty:
            raise ValueError(
                f"Fold {fold_i + 1}: no valid training rows after embargo / NaNs "
                f"(train_end={train_end.date()})"
            )
        if test.empty:
            raise ValueError(
                f"Fold {fold_i + 1}: no valid test rows "
                f"({test_start.date()} → {test_end.date()})"
            )

        y_train = train[y_cols].astype(float)
        if float(y_train.to_numpy().std()) == 0.0:
            raise ValueError(
                f"Fold {fold_i + 1}: training forward returns have zero variance"
            )

        base = make_base_estimator(config.model_name, random_state=random_state)
        model = MultiOutputRegressor(base)
        model.fit(train[FEATURE_COLUMNS], y_train)
        last_model = model

        def _predict_horizons(frame: pd.DataFrame) -> np.ndarray:
            return np.asarray(model.predict(frame[FEATURE_COLUMNS]), dtype=float)

        train_horizons = _predict_horizons(train)
        test_horizons = _predict_horizons(test)
        train_pred_ret = train_horizons.max(axis=1)
        test_pred_ret = test_horizons.max(axis=1)
        train_pred_hold = train_horizons.argmax(axis=1) + 1
        test_pred_hold = test_horizons.argmax(axis=1) + 1

        train_signal = train_pred_ret >= config.entry_min_return
        test_signal = test_pred_ret >= config.entry_min_return

        y_train_best = y_train.max(axis=1)
        train_metrics = _fold_metrics(
            y_train_best,
            train_pred_ret,
            train_pred_hold,
            train_signal,
            n_horizons=len(y_cols),
            embargo_days=config.max_holding_days,
            train_label_end=train_label_end,
        )

        test_labeled = test.dropna(subset=y_cols)
        if len(test_labeled) > 0:
            y_test_best = test_labeled[y_cols].astype(float).max(axis=1)
            labeled_horizons = _predict_horizons(test_labeled)
            pred_test_labeled = labeled_horizons.max(axis=1)
            hold_test_labeled = labeled_horizons.argmax(axis=1) + 1
            signal_labeled = pred_test_labeled >= config.entry_min_return
            test_metrics = _fold_metrics(
                y_test_best,
                pred_test_labeled,
                hold_test_labeled,
                signal_labeled,
                n_horizons=len(y_cols),
                embargo_days=config.max_holding_days,
            )
            oos_y_parts.append(y_test_best)
            oos_pred_parts.append(pred_test_labeled)
            oos_signal_parts.append(signal_labeled)
            oos_hold_parts.append(hold_test_labeled.astype(float))
        else:
            test_metrics = {
                "n_rows": float(len(test)),
                "signal_rate": float(np.mean(test_signal)),
                "mean_predicted_return": float(np.mean(test_pred_ret)),
                "mean_predicted_hold_days": float(np.mean(test_pred_hold)),
                "n_horizons": float(len(y_cols)),
                "embargo_days": float(config.max_holding_days),
                "mae": float("nan"),
                "r2": float("nan"),
                "mean_forward_return": float("nan"),
            }

        scored.loc[test.index, SCORE_COL] = test_pred_ret
        scored.loc[test.index, HOLD_COL] = test_pred_hold.astype(float)
        scored.loc[test.index, "entry_signal"] = test_signal

        if fold_i == 0:
            first_train_metrics = train_metrics
            scored.loc[train.index, SCORE_COL] = train_pred_ret
            scored.loc[train.index, HOLD_COL] = train_pred_hold.astype(float)
            scored.loc[train.index, "entry_signal_is"] = train_signal

        importance_frames.append(
            _permutation_importance(model, train, y_cols, random_state=random_state)
        )
        fold_results.append(
            WalkForwardFoldResult(
                train_end=pd.Timestamp(train_end),
                test_start=pd.Timestamp(test_start),
                test_end=pd.Timestamp(test_end),
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )
        )

    if last_model is None or first_train_metrics is None:
        raise ValueError("Walk-forward produced no fitted folds")

    if oos_y_parts:
        y_oos = pd.concat(oos_y_parts)
        pred_oos = np.concatenate(oos_pred_parts)
        signal_oos = np.concatenate(oos_signal_parts)
        hold_oos = np.concatenate(oos_hold_parts)
        agg_test_metrics = _fold_metrics(
            y_oos,
            pred_oos,
            hold_oos,
            signal_oos,
            n_horizons=len(y_cols),
            embargo_days=config.max_holding_days,
        )
    else:
        agg_test_metrics = {
            "n_rows": 0.0,
            "signal_rate": float("nan"),
            "mean_predicted_return": float("nan"),
            "mean_predicted_hold_days": float("nan"),
            "n_horizons": float(len(y_cols)),
            "embargo_days": float(config.max_holding_days),
            "mae": float("nan"),
            "r2": float("nan"),
            "mean_forward_return": float("nan"),
        }
    agg_test_metrics["n_folds"] = float(n_folds)

    if importance_frames:
        feature_importance = (
            pd.concat(importance_frames, axis=1)
            .mean(axis=1)
            .rename("importance")
            .sort_values(ascending=False)
        )
    else:
        feature_importance = pd.Series(0.0, index=FEATURE_COLUMNS, name="importance")

    first_train_end, first_test_start, _ = fold_windows[0]
    return WalkForwardModelResult(
        model=last_model,
        feature_columns=list(FEATURE_COLUMNS),
        train_end=pd.Timestamp(first_train_end),
        test_start=pd.Timestamp(first_test_start),
        train_metrics=first_train_metrics,
        test_metrics=agg_test_metrics,
        feature_importance=feature_importance,
        panel=scored,
        folds=fold_results,
    )
