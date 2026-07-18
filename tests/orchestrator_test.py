from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from equitylab.screening.orchestrator import run_screen
from equitylab.screening.post_yahoo import ScreenConfig
from equitylab.screening.pre_yahoo import UniverseConfig


def test_run_screen_returns_first_qualifiers_in_universe_order() -> None:
    quotes = [
        {"symbol": "AAA", "shortName": "AAA", "marketCap": 1, "averageDailyVolume3Month": 3},
        {"symbol": "BBB", "shortName": "BBB", "marketCap": 1, "averageDailyVolume3Month": 2},
        {"symbol": "CCC", "shortName": "CCC", "marketCap": 1, "averageDailyVolume3Month": 1},
    ]
    scored = pd.DataFrame(
        {
            "universe_rank": [0, 1, 2],
            "drawdown_52w": [-0.10, -0.95, -0.20],
            "rsi_14": [35.0, 35.0, 30.0],
            "relative_volume_20": [1.5, 1.5, 1.5],
            "distance_from_sma_200": [0.0, 0.0, 0.0],
        },
        index=["AAA", "BBB", "CCC"],
    )

    with (
        patch("equitylab.screening.orchestrator.fetch_universe", return_value=quotes),
        patch("equitylab.screening.orchestrator.score_quotes", return_value=(scored, [])),
    ):
        results, errors = run_screen(
            UniverseConfig(),
            ScreenConfig(max_drawdown_52w=-0.90, max_rsi=40.0, min_relative_volume=1.2),
            start=date(2024, 1, 1),
            end=date(2025, 1, 1),
            max_qualifiers=50,
        )

    assert errors == []
    assert list(results.index) == ["AAA", "CCC"]
    assert "entry_signal" not in results.columns
    assert "signal_score" not in results.columns
