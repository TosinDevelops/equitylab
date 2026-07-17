from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd

from equitylab.screening.post_yahoo import ScreenConfig, score_quotes
from equitylab.screening.pre_yahoo import UniverseConfig, fetch_universe

DEFAULT_MAX_QUALIFIERS = 50


def run_screen(
    universe: UniverseConfig,
    screen: ScreenConfig,
    start: date,
    end: date,
    max_qualifiers: int = DEFAULT_MAX_QUALIFIERS,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch 150 Yahoo candidates; return the first max_qualifiers that pass post-screen."""
    if progress is not None:
        progress(0.0, "Fetching Yahoo universe (150)…")

    quotes = fetch_universe(universe)
    if not quotes:
        return pd.DataFrame(), ["Yahoo EquityQuery returned no tickers."]

    scored, errors = score_quotes(
        quotes,
        screen,
        start,
        end,
        max_qualifiers=max_qualifiers,
        progress=progress,
    )
    if scored.empty:
        return scored, errors

    passes = (
        scored.loc[scored["entry_signal"]]
        .sort_values("universe_rank")
        .head(max_qualifiers)
    )
    if passes.empty:
        # Nothing qualified — return what was scored for UI inspection.
        return scored.sort_values("universe_rank"), errors
    return passes, errors
