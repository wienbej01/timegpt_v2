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
        """Get the entry signal."""
        costs_bps = self.costs.get_costs_bps(symbol, tick_size, last_price)
        costs_abs = (costs_bps / 10_000) * last_price

        base_sigma = max(sigma_5m, 1e-9)
        distance = abs(q50 - last_price)
        threshold = params.k_sigma * base_sigma
        if distance < threshold:
            return 0.0  # Threshold not met

        raw_size = distance / base_sigma
        size = float(max(0.0, min(raw_size, 1.0)))

        if q25 > last_price + costs_abs:
            return size
        if q75 < last_price - costs_abs:
            return -size
        return 0.0  # No signal

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

        if position == 1:  # Long position
            if current_price >= entry_price + params.s_take * sigma_5m:
                return True  # Take profit
            if current_price <= entry_price - params.s_stop * sigma_5m:
                return True  # Stop loss
        elif position == -1:  # Short position
            if current_price <= entry_price - params.s_take * sigma_5m:
                return True  # Take profit
            if current_price >= entry_price + params.s_stop * sigma_5m:
                return True  # Stop loss
        return False
