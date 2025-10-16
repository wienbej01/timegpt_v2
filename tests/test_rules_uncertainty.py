from __future__ import annotations

import datetime
from timegpt_v2.trading.costs import TradingCosts
from timegpt_v2.trading.rules import RuleParams, TradingRules


def test_uncertainty_gate_relaxes():
    rules = TradingRules(
        costs=TradingCosts(fee_bps=0.5, half_spread_ticks={}),
        time_stop=datetime.time(15, 55),
        daily_trade_cap=1,
        max_open_per_symbol=1,
    )
    # sigma = 0.01, spread = 0.025
    params_tight = RuleParams(k_sigma=0.0, s_stop=10.0, s_take=10.0, uncertainty_k=2.0)
    params_loose = RuleParams(k_sigma=0.0, s_stop=10.0, s_take=10.0, uncertainty_k=4.0)
    quantiles = {0.25: 0.001, 0.5: 0.02, 0.75: 0.026}  # spread=0.025
    # Tight gate rejects: 0.025 > 2.0*0.01
    assert rules.get_entry_signal(params_tight, quantiles, 100.0, 0.01, 0.01, "X") == 0.0
    # Looser gate allows: 0.025 <= 4.0*0.01
    assert rules.get_entry_signal(params_loose, quantiles, 100.0, 0.01, 0.01, "X") != 0.0