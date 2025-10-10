from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timegpt_v2.backtest.grid import GridSearch


@pytest.fixture
def synthetic_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return aligned forecasts, features, and prices for the sweep tests."""
    start_ts = pd.Timestamp("2024-01-02 14:30", tz="UTC")
    index = pd.date_range(start_ts, periods=20, freq="1min", tz="UTC")

    close_path = pd.Series(
        [
            100.00,
            100.02,
            100.05,
            100.30,
            100.32,
            100.60,
            100.62,
            100.90,
            100.92,
            101.10,
            101.12,
            101.35,
            101.40,
            101.55,
            101.80,
            101.82,
            101.60,
            101.58,
            101.55,
            101.50,
        ],
        index=index,
    )

    prices = pd.DataFrame(
        {
            "timestamp": index,
            "symbol": "SYN",
            "close": close_path.to_numpy(),
        }
    )

    features = pd.DataFrame(
        {
            "timestamp": index,
            "symbol": "SYN",
            "rv_5m": np.full(len(index), 0.04),
        }
    )

    snapshot_positions = [3, 5]
    forecasts_rows = []
    for pos in snapshot_positions:
        ts = index[pos]
        last_price = float(close_path.iloc[pos])
        forecasts_rows.append(
            {
                "ts_utc": ts.isoformat(),
                "symbol": "SYN",
                "q25": last_price + 0.03,
                "q50": last_price + 0.12,
                "q75": last_price + 0.20,
            }
        )

    forecasts = pd.DataFrame(forecasts_rows)
    return forecasts, features, prices


def test_grid_search_outputs_unique_hashes(
    tmp_path: Path, synthetic_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> None:
    """Each grid point must produce its own summary file and distinct metrics."""

    forecasts, features, prices = synthetic_inputs
    trading_cfg = {
        "k_sigma": [0.5, 1.0],
        "s_stop": [1.0],
        "s_take": [1.0],
        "time_stop_et": "15:55",
        "fees_bps": 0.5,
        "half_spread_ticks": {"SYN": 1},
        "daily_trade_cap": 3,
        "max_open_per_symbol": 1,
    }

    logger = logging.getLogger("test.grid")
    grid_search = GridSearch(
        trading_cfg=trading_cfg,
        logger=logger,
        output_root=tmp_path,
        tick_size=0.01,
    )

    results = grid_search.run(forecasts, features, prices)

    assert len(results) == 2
    assert results["combo_hash"].is_unique

    summary_paths = [
        tmp_path / combo_hash / "bt_summary.csv" for combo_hash in results["combo_hash"]
    ]
    for path in summary_paths:
        assert path.exists(), f"Expected summary at {path}"
