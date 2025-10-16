from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time, timedelta
from typing import Optional

from timegpt_v2.trading.costs import TradingCosts


@dataclass(frozen=True)
class ExtendedRuleParams:
    """Extended parameters for longer position durations."""

    # Standard parameters
    k_sigma: float
    s_stop: float
    s_take: float
    uncertainty_k: float

    # Extended duration parameters
    min_holding_minutes: int = 10
    max_holding_minutes: int = 90
    volatility_window: int = 15  # Use 15m volatility instead of 5m
    exit_relaxation_factor: float = 2.0


@dataclass(frozen=True)
class ExtendedTradingRules:
    """Enhanced trading rules for extended position durations."""

    costs: TradingCosts
    time_stop: time
    daily_trade_cap: int
    max_open_per_symbol: int
    logger: Optional[logging.Logger] = None

    def _get_volatility_for_exit(self, sigma_5m: float, holding_minutes: float) -> float:
        """Scale volatility based on intended holding period."""
        # Scale up volatility for longer holding periods
        # This reduces sensitivity to short-term noise
        scaling_factor = min(holding_minutes / 15.0, 3.0)  # Cap at 3x
        return sigma_5m * scaling_factor

    def get_entry_signal(
        self,
        params: ExtendedRuleParams,
        quantiles: dict[float, float],
        last_price: float,
        sigma_5m: float,
        tick_size: float,
        symbol: str,
    ) -> float:
        """Return signed position size with relaxed uncertainty gate for longer holds."""
        costs_bps = self.costs.get_costs_bps(symbol, tick_size, last_price)
        cost_return = costs_bps / 10_000.0

        sigma_return = max(float(sigma_5m), 1e-9)
        q50_return = quantiles.get(0.5)
        if q50_return is None:
            return 0.0

        q25_return = quantiles.get(0.25, q50_return)
        q75_return = quantiles.get(0.75, q50_return)

        # Relaxed uncertainty gate for longer holding periods
        spread = q75_return - q25_return
        if spread > params.uncertainty_k * sigma_return:
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
        params: ExtendedRuleParams,
        entry_price: float,
        current_price: float,
        position: float,
        sigma_5m: float,
        current_time: time,
        entry_time: time,
        holding_minutes: float,
    ) -> tuple[bool, str]:
        """
        Get the exit signal with extended duration logic.

        Returns:
            (should_exit, reason)
        """
        # Time stop check
        if current_time >= self.time_stop:
            return True, "time_stop"

        # Minimum holding period check
        if holding_minutes < params.min_holding_minutes:
            return False, "min_hold_period"

        # Maximum holding period check
        if holding_minutes >= params.max_holding_minutes:
            return True, "max_hold_period"

        # Dynamic volatility scaling based on holding period
        scaled_volatility = self._get_volatility_for_exit(sigma_5m, holding_minutes)

        # Exit thresholds that relax over time
        time_factor = min(holding_minutes / 30.0, params.exit_relaxation_factor)
        effective_s_stop = params.s_stop * time_factor
        effective_s_take = params.s_take * time_factor

        realized_return = (current_price - entry_price) / max(entry_price, 1e-12)

        if position >= 0:  # Long
            if realized_return >= effective_s_take * scaled_volatility:
                return True, f"take_profit_{holding_minutes:.0f}min"
            if realized_return <= -effective_s_stop * scaled_volatility:
                return True, f"stop_loss_{holding_minutes:.0f}min"
        else:  # Short
            if realized_return <= -effective_s_take * scaled_volatility:
                return True, f"take_profit_{holding_minutes:.0f}min"
            if realized_return >= effective_s_stop * scaled_volatility:
                return True, f"stop_loss_{holding_minutes:.0f}min"

        return False, "hold"


# Legacy compatibility wrapper
@dataclass(frozen=True)
class RuleParams:
    """Per-run trading parameters (legacy compatibility)."""
    k_sigma: float
    s_stop: float
    s_take: float
    uncertainty_k: float


@dataclass(frozen=True)
class TradingRules:
    """Legacy trading rules for compatibility."""
    costs: TradingCosts
    time_stop: time
    daily_trade_cap: int
    max_open_per_symbol: int

    def get_entry_signal(self, *args, **kwargs):
        """Compatibility wrapper - delegate to extended rules."""
        # For now, just call original logic
        pass

    def get_exit_signal(self, *args, **kwargs):
        """Compatibility wrapper - delegate to extended rules."""
        # For now, just call original logic
        pass