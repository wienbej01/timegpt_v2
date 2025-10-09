from __future__ import annotations

from datetime import time

import pytest

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


def test_long_entry_signal(trading_rules: TradingRules) -> None:
    """Test the long entry signal."""
    signal = trading_rules.get_entry_signal(
        q25=101.0,
        q50=102.0,
        q75=103.0,
        last_price=100.0,
        sigma_5m=1.0,
        tick_size=0.01,
        symbol="AAPL",
    )
    assert signal == 1


def test_short_entry_signal(trading_rules: TradingRules) -> None:
    """Test the short entry signal."""
    signal = trading_rules.get_entry_signal(
        q25=97.0,
        q50=98.0,
        q75=99.0,
        last_price=100.0,
        sigma_5m=1.0,
        tick_size=0.01,
        symbol="AAPL",
    )
    assert signal == -1


def test_no_entry_signal(trading_rules: TradingRules) -> None:
    """Test that no entry signal is generated when conditions are not met."""
    signal = trading_rules.get_entry_signal(
        q25=100.0,
        q50=100.0,
        q75=100.0,
        last_price=100.0,
        sigma_5m=1.0,
        tick_size=0.01,
        symbol="AAPL",
    )
    assert signal == 0


def test_long_exit_signal_take_profit(trading_rules: TradingRules) -> None:
    """Test the long exit signal for take profit."""
    exit_signal = trading_rules.get_exit_signal(
        entry_price=100.0,
        current_price=101.0,
        position=1,
        sigma_5m=1.0,
        current_time=time(10, 0),
    )
    assert exit_signal is True


def test_long_exit_signal_stop_loss(trading_rules: TradingRules) -> None:
    """Test the long exit signal for stop loss."""
    exit_signal = trading_rules.get_exit_signal(
        entry_price=100.0,
        current_price=99.0,
        position=1,
        sigma_5m=1.0,
        current_time=time(10, 0),
    )
    assert exit_signal is True


def test_short_exit_signal_take_profit(trading_rules: TradingRules) -> None:
    """Test the short exit signal for take profit."""
    exit_signal = trading_rules.get_exit_signal(
        entry_price=100.0,
        current_price=99.0,
        position=-1,
        sigma_5m=1.0,
        current_time=time(10, 0),
    )
    assert exit_signal is True


def test_short_exit_signal_stop_loss(trading_rules: TradingRules) -> None:
    """Test the short exit signal for stop loss."""
    exit_signal = trading_rules.get_exit_signal(
        entry_price=100.0,
        current_price=101.0,
        position=-1,
        sigma_5m=1.0,
        current_time=time(10, 0),
    )
    assert exit_signal is True


def test_time_stop_exit_signal(trading_rules: TradingRules) -> None:
    """Test the time stop exit signal."""
    exit_signal = trading_rules.get_exit_signal(
        entry_price=100.0,
        current_price=100.0,
        position=1,
        sigma_5m=1.0,
        current_time=time(16, 0),
    )
    assert exit_signal is True
