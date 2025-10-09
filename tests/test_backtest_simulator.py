from __future__ import annotations

import logging
from datetime import time

import pandas as pd
import pytest

from timegpt_v2.backtest.simulator import BacktestSimulator
from timegpt_v2.trading.costs import TradingCosts
from timegpt_v2.trading.rules import TradingRules


@pytest.fixture
def trading_costs() -> TradingCosts:
    """Return a TradingCosts instance for testing."""
    return TradingCosts(fee_bps=0.5, half_spread_ticks={"AAPL": 1})


@pytest.fixture
def trading_rules(trading_costs: TradingCosts) -> TradingRules:
    """Return a TradingRules instance for testing."""
    return TradingRules(
        costs=trading_costs,
        k_sigma=0.5,
        s_stop=1.0,
        s_take=1.0,
        time_stop=time(15, 55),
        daily_trade_cap=3,
        max_open_per_symbol=1,
    )


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


def test_backtest_simulator_run(
    trading_rules: TradingRules,
    forecasts: pd.DataFrame,
    features: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:
    """Test the run method of the BacktestSimulator."""
    logger = logging.getLogger(__name__)
    simulator = BacktestSimulator(rules=trading_rules, logger=logger)
    trades, summary = simulator.run(forecasts, features, prices)

    assert len(trades) == 1
    assert summary["total_pnl"].iloc[0] > 0
    assert summary["trade_count"].iloc[0] == 1
