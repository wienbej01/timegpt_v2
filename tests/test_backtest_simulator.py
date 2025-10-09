from __future__ import annotations

import logging
from datetime import time

import numpy as np
import pandas as pd
import pytest

from timegpt_v2.backtest.simulator import BacktestSimulator
from timegpt_v2.fe.base_features import build_feature_matrix
from timegpt_v2.trading.costs import TradingCosts
from timegpt_v2.trading.rules import RuleParams, TradingRules
from timegpt_v2.utils.synthetic import SyntheticConfig, generate_bars


def test_backtest_simulator_cashflow_accounting() -> None:
    """Simulator should produce cashflows that reconcile with trade blotter."""
    bars = generate_bars(SyntheticConfig(symbol="SYN", minutes_per_session=30, seed=11))
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("UTC")
    features = build_feature_matrix(bars)[["timestamp", "symbol", "rv_5m"]].copy()
    prices = bars[["timestamp", "symbol", "close"]].copy()
    prices["timestamp"] = prices["timestamp"].dt.tz_convert("UTC")

    snapshot_row = features.iloc[5]
    snapshot_ts = snapshot_row["timestamp"]
    rv_value = float(snapshot_row["rv_5m"])
    if rv_value <= 0:
        features.loc[features["timestamp"] == snapshot_ts, "rv_5m"] = 0.04
        sigma = 0.2
    else:
        sigma = float(np.sqrt(rv_value))
    price_at_snapshot = float(
        prices.loc[
            (prices["timestamp"] == snapshot_ts) & (prices["symbol"] == snapshot_row["symbol"]),
            "close",
        ].iloc[0]
    )
    forecasts = pd.DataFrame(
        {
            "ts_utc": [snapshot_ts.isoformat()],
            "symbol": [snapshot_row["symbol"]],
            "q25": [price_at_snapshot + max(0.02, sigma * 0.6)],
            "q50": [price_at_snapshot + max(0.03, sigma * 0.7)],
            "q75": [price_at_snapshot + max(0.04, sigma * 0.8)],
        }
    )

    trading_costs = TradingCosts(fee_bps=0.5, half_spread_ticks={"SYN": 1})
    trading_rules = TradingRules(
        costs=trading_costs,
        time_stop=time(15, 55),
        daily_trade_cap=3,
        max_open_per_symbol=1,
    )
    params = RuleParams(k_sigma=0.5, s_stop=1.0, s_take=1.0)
    logger = logging.getLogger("test.backtest")
    simulator = BacktestSimulator(
        rules=trading_rules,
        params=params,
        logger=logger,
        tick_size=0.01,
    )

    trades, summary = simulator.run(forecasts, features, prices)

    assert not trades.empty, "Expected at least one trade"
    assert not trades.isna().any().any(), "Trade blotter should not contain NaNs"
    total_net = trades["net_pnl"].sum()
    assert summary["trade_count"].iloc[0] == len(trades)
    assert pytest.approx(summary["total_net_pnl"].iloc[0], rel=1e-6) == total_net
