from __future__ import annotations

import logging

import pandas as pd
import pytest

from timegpt_v2.backtest.grid import GridSearch


@pytest.fixture
def trading_cfg() -> dict:
    """Return a sample trading config for testing."""
    return {
        "k_sigma": [0.5, 1.0],
        "s_stop": [1.0, 1.5],
        "s_take": [1.0],
        "time_stop_et": "15:55",
        "fees_bps": 0.5,
        "half_spread_ticks": {"AAPL": 1},
        "daily_trade_cap": 3,
        "max_open_per_symbol": 1,
    }


@pytest.fixture
def forecasts() -> pd.DataFrame:
    """Return a sample forecasts DataFrame for testing."""
    return pd.DataFrame(
        {
            "snapshot_utc": pd.to_datetime(["2024-07-01 10:00:00"]),
            "symbol": ["AAPL"],
            "q25": [101.0],
            "q50": [102.0],
            "q75": [103.0],
        }
    )


@pytest.fixture
def features() -> pd.DataFrame:
    """Return a sample features DataFrame for testing."""
    index = pd.to_datetime(
        [
            "2024-07-01 10:00:00",
            "2024-07-01 10:01:00",
            "2024-07-01 10:02:00",
        ],
        utc=True,
    )
    return pd.DataFrame({"rv_5m": [1.0, 1.0, 1.0]}, index=index)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Return a sample prices DataFrame for testing."""
    index = pd.to_datetime(
        [
            "2024-07-01 09:59:00",
            "2024-07-01 10:00:00",
            "2024-07-01 10:01:00",
            "2024-07-01 10:02:00",
        ],
        utc=True,
    )
    return pd.DataFrame({"AAPL": [99.0, 100.0, 101.0, 102.0]}, index=index)


def test_grid_search_run(
    trading_cfg: dict,
    forecasts: pd.DataFrame,
    features: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Test the run method of the GridSearch."""
    logger = logging.getLogger(__name__)
    grid_search = GridSearch(trading_cfg=trading_cfg, logger=logger)
    results = grid_search.run(forecasts, features, prices)

    assert len(results) == 4
    assert len(results["grid_point"].unique()) == 4
