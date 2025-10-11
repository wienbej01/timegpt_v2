from __future__ import annotations

from dataclasses import dataclass
from datetime import time

from timegpt_v2.trading.costs import TradingCosts


@dataclass(frozen=True)
class RuleParams:
    """Per-run trading parameters swept by the grid."""

    k_sigma: float
    s_stop: float
    s_take: float


@dataclass(frozen=True)
class TradingRules:
    """Configuration for trading rules."""

    costs: TradingCosts
    time_stop: time
    daily_trade_cap: int
    max_open_per_symbol: int

    def get_entry_signal(
        self,
        params: RuleParams,
        q25: float,
        q50: float,
        q75: float,
        last_price: float,
        sigma_5m: float,
        tick_size: float,
        symbol: str,
    ) -> float:
        """Return signed position size in units of the maximum clip."""
        costs_bps = self.costs.get_costs_bps(symbol, tick_size, last_price)
        cost_return = costs_bps / 10_000.0

        sigma_return = max(float(sigma_5m), 1e-9)
        q50_return = float(q50)
        
        q25_return = float(q25)
        q75_return = float(q75)

        # Uncertainty suppression: wide intervals indicate high uncertainty
        spread = q75_return - q25_return
        if spread > 2.0 * sigma_return:
            return 0.0

        threshold = params.k_sigma * sigma_return
        if abs(q50_return) < threshold:
            return 0.0

        signal = 0.0
        if q25_return > cost_return:
            signal = 1.0
        elif q75_return < -cost_return:
            signal = -1.0

        # EV(after-cost) > 0 check
        if signal > 0 and q50_return - cost_return <= 0:
            return 0.0
        if signal < 0 and q50_return + cost_return >= 0:
            return 0.0

        return signal

    def get_exit_signal(
        self,
        params: RuleParams,
        entry_price: float,
        current_price: float,
        position: float,
        sigma_5m: float,
        current_time: time,
    ) -> bool:
        """Get the exit signal."""
        if current_time >= self.time_stop:
            return True

        sigma_return = max(float(sigma_5m), 1e-9)
        realized_return = (current_price - entry_price) / max(entry_price, 1e-12)

        if position >= 0:  # Long
            if realized_return >= params.s_take * sigma_return:
                return True
            if realized_return <= -params.s_stop * sigma_return:
                return True
        else:  # Short
            if realized_return <= -params.s_take * sigma_return:
                return True
            if realized_return >= params.s_stop * sigma_return:
                return True
        return False
