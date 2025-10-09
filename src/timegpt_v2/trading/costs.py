from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class TradingCosts:
    """Configuration for trading costs."""

    fee_bps: float
    half_spread_ticks: Mapping[str, float]

    def get_costs_bps(self, symbol: str, tick_size: float, price: float) -> float:
        """Calculate the total costs in basis points."""
        half_spread = self.half_spread_ticks.get(symbol, 0.0) * tick_size
        half_spread_bps = (half_spread / price) * 10_000
        return self.fee_bps + half_spread_bps
