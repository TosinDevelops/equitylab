from datetime import date, timedelta

import pandas as pd
import streamlit as st

from equitylab.data.loaders.yahoo import load_data
from equitylab.screening import ScreenConfig, UniverseConfig, run_screen
from equitylab.strategy import StrategyConfig, run_walkforward_strategy

st.set_page_config(page_title="EquityLab", layout="wide")
st.title("EquityLab")
st.caption("Screen stocks, then chart any match over the selected date range.")

today = date.today()
default_start = today - timedelta(days=365)

if "screen_results" not in st.session_state:
    st.session_state.screen_results = None
if "screen_errors" not in st.session_state:
    st.session_state.screen_errors = []
if "screen_label" not in st.session_state:
    st.session_state.screen_label = ""
if "screen_range" not in st.session_state:
    st.session_state.screen_range = (default_start, today)
if "strategy_result" not in st.session_state:
    st.session_state.strategy_result = None
# Drop cached results built with an older StrategyConfig (missing newer fields).
_cached = st.session_state.strategy_result
if _cached is not None and (
    not hasattr(_cached.config, "profit_drawdown")
    or not hasattr(_cached.config, "model_horizon_exit")
):
    st.session_state.strategy_result = None

with st.sidebar:
    st.header("Date range")
    start_date, end_date = st.date_input(
        "Range",
        value=st.session_state.screen_range,
        max_value=today,
    )

    st.header("Universe (Yahoo)")
    st.caption("Always fetches 150 Yahoo candidates (volume-sorted). Post-screen keeps the first N that qualify.")
    max_tickers = st.number_input(
        "Max tickers",
        min_value=1,
        max_value=50,
        value=50,
        step=5,
        help="After post-filters, keep the first N qualifiers (in Yahoo volume order). Cap 50.",
    )
    min_market_cap_b = st.number_input("Min market cap ($B)", min_value=0.0, value=0.5, step=0.1)
    max_market_cap_b = st.number_input(
        "Max market cap ($B, 0 = none)",
        min_value=0.0,
        value=100.0,
        step=1.0,
    )
    min_price = st.number_input("Min price ($)", min_value=0.0, value=5.0, step=0.5)
    min_avg_vol = st.number_input("Min avg daily volume (3m)", min_value=0, value=500_000, step=50_000)

    st.header("Screen")
    max_drawdown_pct = st.slider(
        "Max 52w drawdown",
        min_value=0,
        max_value=90,
        value=90,
        step=5,
        format="%d%%",
        help="Allow names down by at most this much from the 52-week high. 90% includes 50%, 10%, etc.",
    )
    max_rsi = st.slider("Max RSI (14)", min_value=0.0, max_value=100.0, value=40.0, step=1.0)
    min_rel_vol = st.slider("Min relative volume (20d)", min_value=0.5, max_value=5.0, value=1.2, step=0.1)
    use_sma_band = st.checkbox("Filter distance from 200D SMA", value=False)
    min_sma_dist = None
    max_sma_dist = None
    if use_sma_band:
        min_sma_dist = st.number_input("Min distance from SMA200", value=-0.20, step=0.05, format="%.2f")
        max_sma_dist = st.number_input("Max distance from SMA200", value=0.05, step=0.05, format="%.2f")

    run = st.button("Run screen", type="primary", width="stretch")

    st.header("Strategy / Backtest")
    st.caption(
        "Walk-forward predicts buy→sell profit over max holding days (train 80%, trade 20%)."
    )
    max_positions = st.number_input("Max positions", min_value=1, max_value=20, value=5, step=1)
    max_holding_days = st.number_input("Max holding days", min_value=1, max_value=60, value=20, step=1)
    train_fraction = st.slider("Train fraction", min_value=0.5, max_value=0.9, value=0.80, step=0.05)
    entry_min_return_pct = st.number_input(
        "Entry min predicted return (%)",
        min_value=-5.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        help="Enter when model predicts N-day buy→sell return at least this high.",
    )
    model_horizon_exit = st.checkbox(
        "Exit on model horizon",
        value=False,
        help="Exit at the hold day the model predicts is best (argmax of 1..N returns).",
    )
    use_profit_drawdown = st.checkbox("Use profit drawdown exit", value=True)
    profit_drawdown_pct = 5.0
    if use_profit_drawdown:
        profit_drawdown_pct = st.number_input(
            "Profit drawdown exit (%)",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            help="Exit when close falls this far below the peak price since entry.",
        )
    cost_bps = st.number_input("Cost (bps per side)", min_value=0.0, value=5.0, step=1.0)
    use_stops = st.checkbox("Use stop / take-profit", value=False)
    stop_loss = None
    take_profit = None
    if use_stops:
        stop_loss = st.number_input("Stop loss", value=-0.08, step=0.01, format="%.2f")
        take_profit = st.number_input("Take profit", value=0.15, step=0.01, format="%.2f")
    run_wf = st.button("Run walk-forward", width="stretch")

if not isinstance(start_date, date) or not isinstance(end_date, date):
    st.error("Select both a start and end date.")
    st.stop()

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()

if end_date - start_date < timedelta(days=200):
    st.warning("Range is short for SMA200 / 52w stats — screener still extends lookback automatically.")

if run:
    universe = UniverseConfig(
        min_market_cap=min_market_cap_b * 1_000_000_000,
        max_market_cap=(max_market_cap_b * 1_000_000_000) if max_market_cap_b > 0 else None,
        min_price=float(min_price),
        min_avg_daily_volume=float(min_avg_vol),
    )
    screen = ScreenConfig(
        max_drawdown_52w=-float(max_drawdown_pct) / 100.0,  # UI is positive %; filter uses signed drawdown
        max_rsi=float(max_rsi),
        min_relative_volume=float(min_rel_vol),
        min_distance_from_sma_200=float(min_sma_dist) if min_sma_dist is not None else None,
        max_distance_from_sma_200=float(max_sma_dist) if max_sma_dist is not None else None,
    )

    progress = st.progress(0.0, text="Fetching Yahoo universe…")

    def on_progress(fraction: float, message: str) -> None:
        progress.progress(min(max(fraction, 0.0), 1.0), text=message)

    try:
        results, errors = run_screen(
            universe,
            screen,
            start=start_date,
            end=end_date,
            max_qualifiers=int(max_tickers),
            progress=on_progress,
        )
    except Exception as exc:
        progress.empty()
        st.error(f"Screen failed: {exc}")
        st.stop()

    progress.empty()
    st.session_state.screen_results = results
    st.session_state.screen_errors = errors
    st.session_state.screen_label = screen.label
    st.session_state.screen_range = (start_date, end_date)
    st.session_state.selected_ticker = None

results = st.session_state.screen_results
if results is None:
    st.info("Configure filters in the sidebar, then click **Run screen**.")
    st.stop()

if results.empty:
    st.warning("No tickers scored. Try loosening universe filters.")
    if st.session_state.screen_errors:
        with st.expander("Errors"):
            st.write(st.session_state.screen_errors)
    st.stop()

st.subheader("Screen results")
st.caption(st.session_state.screen_label)

display = results if len(results) <= int(max_tickers) else results.head(int(max_tickers))
tickers = list(display.index)

show_cols = [
    c
    for c in [
        "name",
        "close",
        "drawdown_52w",
        "rsi_14",
        "relative_volume_20",
        "distance_from_sma_200",
        "market_cap",
    ]
    if c in display.columns
]
st.dataframe(display[show_cols], width="stretch")

if st.session_state.screen_errors:
    with st.expander(f"Skipped / errors ({len(st.session_state.screen_errors)})"):
        st.write(st.session_state.screen_errors)

default_ticker = st.session_state.get("selected_ticker")
if default_ticker not in tickers:
    default_ticker = tickers[0]

selected = st.selectbox(
    "Chart ticker",
    options=tickers,
    index=tickers.index(default_ticker),
)
st.session_state.selected_ticker = selected

chart_start, chart_end = st.session_state.screen_range
try:
    with st.spinner(f"Loading {selected}…"):
        prices = load_data(
            selected,
            interval="1d",
            start=chart_start.isoformat(),
            end=chart_end.isoformat(),
        )
except Exception as exc:
    st.error(f"Failed to load {selected}: {exc}")
    st.stop()

st.subheader(selected)
st.write(f"{len(prices)} bars · {prices.index.min().date()} → {prices.index.max().date()}")
st.line_chart(prices["close"], height=320)
st.dataframe(prices, width="stretch")

if run_wf:
    strategy = StrategyConfig(
        max_positions=int(max_positions),
        max_holding_days=int(max_holding_days),
        train_fraction=float(train_fraction),
        entry_min_return=float(entry_min_return_pct) / 100.0,
        exit_min_return=None,
        profit_drawdown=(float(profit_drawdown_pct) / 100.0) if use_profit_drawdown else None,
        model_horizon_exit=bool(model_horizon_exit),
        cost_bps=float(cost_bps),
        stop_loss=float(stop_loss) if stop_loss is not None else None,
        take_profit=float(take_profit) if take_profit is not None else None,
    )
    wf_progress = st.progress(0.0, text="Starting walk-forward…")

    def on_wf_progress(fraction: float, message: str) -> None:
        wf_progress.progress(min(max(fraction, 0.0), 1.0), text=message)

    try:
        st.session_state.strategy_result = run_walkforward_strategy(
            tickers,
            start=start_date,
            end=end_date,
            config=strategy,
            progress=on_wf_progress,
        )
    except Exception as exc:
        wf_progress.empty()
        st.error(f"Walk-forward failed: {exc}")
        st.stop()
    wf_progress.empty()

strategy_result = st.session_state.strategy_result
if strategy_result is not None:
    st.subheader("Walk-forward strategy (OOS)")
    exit_bits: list[str] = []
    if strategy_result.config.model_horizon_exit:
        exit_bits.append("model horizon exit")
    if strategy_result.config.profit_drawdown is not None:
        exit_bits.append(f"trail DD {strategy_result.config.profit_drawdown:.0%}")
    exit_bits.append(f"hold ≤ {strategy_result.config.max_holding_days}d")
    st.caption(
        f"Train through {strategy_result.train_end.date()} · "
        f"Test from {strategy_result.test_start.date()} · "
        f"{len(strategy_result.tickers)} tickers · "
        f"max {strategy_result.config.max_positions} positions · "
        f"enter pred ≥ {strategy_result.config.entry_min_return:.1%} · "
        + " · ".join(exit_bits)
    )

    oos = strategy_result.oos_backtest
    metric_cols = st.columns(4)
    metric_cols[0].metric("OOS total return", f"{oos.metrics['total_return']:.1%}")
    metric_cols[1].metric("OOS Sharpe", f"{oos.metrics['sharpe']:.2f}")
    metric_cols[2].metric("OOS max DD", f"{oos.metrics['max_drawdown']:.1%}")
    metric_cols[3].metric("OOS trades", f"{int(oos.metrics['trade_count'])}")

    st.line_chart(oos.equity, height=280)
    if not oos.trades.empty:
        st.dataframe(oos.trades, width="stretch")
    else:
        st.info("No OOS trades — try lowering entry probability or using more tickers / longer range.")

    with st.expander("Model diagnostics"):
        diag = pd.DataFrame(
            {
                "train": pd.Series(strategy_result.train_metrics),
                "test": pd.Series(strategy_result.test_metrics),
            }
        )
        st.dataframe(diag, width="stretch")
        st.write("Feature importance (permutation, train sample)")
        st.dataframe(strategy_result.feature_importance.to_frame(), width="stretch")
        st.write("In-sample backtest (diagnostic only)")
        is_bt = strategy_result.is_backtest
        st.write(
            f"Return {is_bt.metrics['total_return']:.1%} · "
            f"Sharpe {is_bt.metrics['sharpe']:.2f} · "
            f"Trades {int(is_bt.metrics['trade_count'])}"
        )

    if strategy_result.errors:
        with st.expander(f"Price load errors ({len(strategy_result.errors)})"):
            st.write(strategy_result.errors)
