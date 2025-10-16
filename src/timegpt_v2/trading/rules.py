from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import pandas as pd

from timegpt_v2.trading.costs import TradingCosts
from timegpt_v2.utils.sigma_utils import compute_horizon_sigma, validate_tp_sl_geometry


@dataclass(frozen=True)
class RuleParams:
    """Per-run trading parameters swept by the grid."""

    k_sigma: float
    s_stop: float
    s_take: float
    uncertainty_k: float
    horizon_minutes: int = 30  # Forecast horizon for sigma alignment
    sigma_basis: str = "horizon"  # "horizon" or "5m" for sigma calculation


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
        quantiles: dict[float, float],
        last_price: float,
        sigma_5m: float,
        tick_size: float,
        symbol: str,
        features: Optional[pd.DataFrame] = None,
        entry_time: Optional[pd.Timestamp] = None,
    ) -> float:
        """Return signed position size in units of the maximum clip."""
        costs_bps = self.costs.get_costs_bps(symbol, tick_size, last_price)
        cost_return = costs_bps / 10_000.0

        # Use horizon-aligned sigma if available and configured
        if (params.sigma_basis == "horizon" and features is not None and entry_time is not None):
            try:
                sigma_h = compute_horizon_sigma(
                    features=features,
                    symbol=symbol,
                    timestamp=entry_time,
                    horizon_minutes=params.horizon_minutes,
                    logger=self._get_logger()
                )
                sigma_return = max(float(sigma_h), 1e-9)
            except Exception:
                # Fallback to provided sigma_5m
                sigma_return = max(float(sigma_5m), 1e-9)
        else:
            sigma_return = max(float(sigma_5m), 1e-9)

        q50_return = quantiles.get(0.5)
        if q50_return is None:
            return 0.0

        q25_return = quantiles.get(0.25, q50_return)
        q75_return = quantiles.get(0.75, q50_return)

        # Uncertainty suppression: wide intervals indicate high uncertainty
        spread = q75_return - q25_return
        # configurable (was hard-coded 4.0)
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
        params: RuleParams,
        entry_price: float,
        current_price: float,
        position: float,
        sigma_5m: float,
        current_time: time,
        features: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
        current_timestamp: Optional[pd.Timestamp] = None,
        quantiles: Optional[dict[float, float]] = None,
    ) -> bool:
        """Get the exit signal."""
        if current_time >= self.time_stop:
            return True

        # Validate TP/SL geometry on first use
        if not hasattr(self, '_geometry_validated'):
            validate_tp_sl_geometry(params.s_take, params.s_stop, self._get_logger())
            self._geometry_validated = True

        # Use horizon-aligned sigma if available and configured
        if (params.sigma_basis == "horizon" and features is not None and
            symbol is not None and current_timestamp is not None):
            try:
                sigma_h = compute_horizon_sigma(
                    features=features,
                    symbol=symbol,
                    timestamp=current_timestamp,
                    horizon_minutes=params.horizon_minutes,
                    logger=self._get_logger()
                )
                sigma_return = max(float(sigma_h), 1e-9)
            except Exception:
                # Fallback to provided sigma_5m
                sigma_return = max(float(sigma_5m), 1e-9)
        else:
            sigma_return = max(float(sigma_5m), 1e-9)

        realized_return = (current_price - entry_price) / max(entry_price, 1e-12)

        # Standard TP/SL exit conditions
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

        # EV-based exit check if quantiles are available
        if quantiles is not None and current_timestamp is not None:
            costs_bps = self.costs.get_costs_bps(symbol, 0.01, current_price)
            should_exit, reason = compute_ev_exit_threshold(
                q50=quantiles.get(0.5, 0.0),
                q25=quantiles.get(0.25, 0.0),
                q75=quantiles.get(0.75, 0.0),
                sigma_h=sigma_return,
                costs_bps=costs_bps,
                price=current_price,
                logger=self._get_logger()
            )
            if should_exit:
                return True

        return False

    def _get_logger(self):
        """Get logger instance."""
        import logging
        return logging.getLogger(__name__)
