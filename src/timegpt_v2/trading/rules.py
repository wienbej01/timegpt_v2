from __future__ import annotations

from dataclasses import dataclass
from datetime import time

from timegpt_v2.trading.costs import TradingCosts


@dataclass(frozen=True)
class TradingRules:
    """Configuration for trading rules."""

    costs: TradingCosts
    k_sigma: float
    s_stop: float
    s_take: float
    time_stop: time
    daily_trade_cap: int
    max_open_per_symbol: int

    def get_entry_signal(
        self,
        q25: float,
        q50: float,
        q75: float,
        last_price: float,
        sigma_5m: float,
        tick_size: float,
        symbol: str,
    ) -> int:
        """Get the entry signal."""
        costs_bps = self.costs.get_costs_bps(symbol, tick_size, last_price)
        costs_abs = (costs_bps / 10_000) * last_price

        if q25 > last_price + costs_abs and abs(q50 - last_price) >= self.k_sigma * sigma_5m:
            return 1  # Long
        if q75 < last_price - costs_abs and abs(q50 - last_price) >= self.k_sigma * sigma_5m:
            return -1  # Short
        return 0  # No signal

    def get_exit_signal(
        self,
        entry_price: float,
        current_price: float,
        position: int,
        sigma_5m: float,
        current_time: time,
    ) -> bool:
        """Get the exit signal."""
        if current_time >= self.time_stop:
            return True

        if position == 1:  # Long position
            if current_price >= entry_price + self.s_take * sigma_5m:
                return True  # Take profit
            if current_price <= entry_price - self.s_stop * sigma_5m:
                return True  # Stop loss
        elif position == -1:  # Short position
            if current_price <= entry_price - self.s_take * sigma_5m:
                return True  # Take profit
            if current_price >= entry_price + self.s_stop * sigma_5m:
                return True  # Stop loss
        return False
