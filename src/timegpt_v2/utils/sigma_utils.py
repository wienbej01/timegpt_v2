"""Utilities for horizon-aligned sigma computation."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def compute_horizon_sigma(
    features: pd.DataFrame,
    symbol: str,
    timestamp: pd.Timestamp,
    horizon_minutes: int,
    *,
    default_sigma_col: str = "sigma_5m",
    fallback_cols: list[str] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Compute sigma aligned with forecast horizon.

    Args:
        features: Historical feature matrix with volatility measures
        symbol: Symbol for which to compute sigma
        timestamp: Timestamp at which sigma is needed
        horizon_minutes: Forecast horizon (30 or 60 minutes)
        default_sigma_col: Default sigma column to use as fallback
        fallback_cols: List of alternative sigma columns to try
        logger: Optional logger for diagnostics

    Returns:
        Horizon-aligned sigma value

    Raises:
        ValueError: If no suitable sigma column is found
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if fallback_cols is None:
        fallback_cols = ["rv_5m", "vol_ewm_15m", "vol_ewm_30m", "vol_ewm_60m"]

    # Filter for symbol and timestamp <= requested time
    symbol_data = features[
        (features["symbol"] == symbol) &
        (features["timestamp"] <= timestamp)
    ].copy()

    if symbol_data.empty:
        raise ValueError(f"No data available for {symbol} before {timestamp}")

    # Sort by timestamp to get the most recent value
    symbol_data = symbol_data.sort_values("timestamp", ascending=False)

    # Strategy 1: Use pre-computed horizon sigma if available
    horizon_sigma_col = f"sigma_{horizon_minutes}m"
    if horizon_sigma_col in symbol_data.columns:
        sigma_value = symbol_data[horizon_sigma_col].iloc[0]
        if not pd.isna(sigma_value) and sigma_value > 0:
            logger.debug(
                "Using pre-computed %s for %s at %s: %.6f",
                horizon_sigma_col, symbol, timestamp, sigma_value
            )
            return float(sigma_value)

    # Strategy 2: Use available volatility measure and scale to horizon
    for col in fallback_cols:
        if col in symbol_data.columns:
            sigma_value = symbol_data[col].iloc[0]
            if not pd.isna(sigma_value) and sigma_value > 0:
                # Scale to horizon using sqrt(time) assumption
                if "5m" in col or col == "rv_5m":
                    base_minutes = 5
                elif "15m" in col:
                    base_minutes = 15
                elif "30m" in col:
                    base_minutes = 30
                elif "60m" in col:
                    base_minutes = 60
                else:
                    base_minutes = 5  # Default assumption

                # Scale sigma using square-root-of-time rule
                scaling_factor = np.sqrt(horizon_minutes / base_minutes)
                horizon_sigma = sigma_value * scaling_factor

                logger.info(
                    "Scaled %s (%.1fm base) to %s: %.6f -> %.6f (factor=%.3f)",
                    col, base_minutes, f"{horizon_minutes}m", sigma_value, horizon_sigma, scaling_factor
                )
                return float(horizon_sigma)

    # Strategy 3: Compute from returns if nothing else available
    if "ret_1m" in features.columns:
        # Compute rolling volatility for the horizon
        symbol_returns = features[
            (features["symbol"] == symbol) &
            (features["timestamp"] <= timestamp)
        ].sort_values("timestamp", ascending=False)

        if len(symbol_returns) >= horizon_minutes:
            recent_returns = symbol_returns["ret_1m"].head(horizon_minutes)
            horizon_sigma = recent_returns.std() * np.sqrt(252 * 390)  # Annualized

            if horizon_sigma > 0:
                logger.info(
                    "Computed %s from returns for %s: %.6f (n=%d)",
                    f"{horizon_minutes}m", symbol, horizon_sigma, len(recent_returns)
                )
                return float(horizon_sigma)

    # Fallback: Use default sigma column with simple scaling
    if default_sigma_col in symbol_data.columns:
        sigma_value = symbol_data[default_sigma_col].iloc[0]
        if not pd.isna(sigma_value) and sigma_value > 0:
            # Simple sqrt scaling from 5m to horizon
            scaling_factor = np.sqrt(horizon_minutes / 5)
            horizon_sigma = sigma_value * scaling_factor

            logger.warning(
                "Using fallback scaling from %s to %s: %.6f -> %.6f",
                default_sigma_col, f"{horizon_minutes}m", sigma_value, horizon_sigma
            )
            return float(horizon_sigma)

    raise ValueError(
        f"Could not compute {horizon_minutes}m sigma for {symbol} at {timestamp}. "
        f"Available columns: {list(features.columns)}"
    )


def validate_tp_sl_geometry(tp_sigma: float, sl_sigma: float, logger: Optional[logging.Logger] = None) -> None:
    """Validate that TP >= SL for positive expectancy.

    Args:
        tp_sigma: Take-profit distance in sigma units
        sl_sigma: Stop-loss distance in sigma units
        logger: Optional logger

    Raises:
        ValueError: If TP < SL (negative expectancy geometry)
    """
    if tp_sigma < sl_sigma:
        error_msg = f"Negative expectancy: TP ({tp_sigma}σ) < SL ({sl_sigma}σ). This requires unrealistically high win rates."
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)

    if logger:
        logger.info("Payoff geometry validated: TP (%.2fσ) >= SL (%.2fσ)", tp_sigma, sl_sigma)


def compute_ev_exit_threshold(
    q50: float,
    q25: float,
    q75: float,
    sigma_h: float,
    costs_bps: float,
    price: float,
    logger: Optional[logging.Logger] = None,
) -> Tuple[bool, str]:
    """Compute expected value-based exit threshold.

    Args:
        q50: Median forecast return
        q25: 25th percentile forecast return
        q75: 75th percentile forecast return
        sigma_h: Horizon-aligned sigma
        costs_bps: Trading costs in basis points
        price: Current price
        logger: Optional logger

    Returns:
        Tuple of (should_exit, reason)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Convert costs to return units
    cost_return = (costs_bps / 10000.0)

    # Expected value based on quantile forecast
    # Simple approximation: EV = median - cost_adjustment
    expected_return = q50 - cost_return

    # Uncertainty penalty based on spread
    uncertainty_penalty = (q75 - q25) / 2.0
    adjusted_ev = expected_return - uncertainty_penalty

    # Exit if EV becomes negative
    if adjusted_ev <= 0:
        logger.debug(
            "EV exit: q50=%.6f, cost=%.6f, uncertainty=%.6f, EV=%.6f <= 0",
            q50, cost_return, uncertainty_penalty, adjusted_ev
        )
        return True, "negative_expected_value"

    # Additional check: if forecast suggests adverse move
    adverse_threshold = -sigma_h * 0.5  # 0.5 sigma adverse move
    if q50 < adverse_threshold:
        logger.debug(
            "EV exit: q50=%.6f < adverse_threshold=%.6f",
            q50, adverse_threshold
        )
        return True, "adverse_forecast"

    return False, "continue_holding"


__all__ = [
    "compute_horizon_sigma",
    "validate_tp_sl_geometry",
    "compute_ev_exit_threshold"
]