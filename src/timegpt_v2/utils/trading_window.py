"""Trading window configuration and validation utilities."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

from timegpt_v2.config.model import TradingWindowConfig


def parse_trading_window_config(
    config: Dict,
    legacy_dates_key: str = "dates",
    legacy_history_days_key: str = "rolling_history_days",
    logger: Optional[logging.Logger] = None,
) -> TradingWindowConfig:
    """
    Parse trading window configuration with backward compatibility.

    Args:
        config: Configuration dictionary that may contain:
               - trading_window: {start, end, history_backfill_days, enforce_trading_window}
               - dates: {start, end} (legacy format)
               - rolling_history_days: int (legacy format)
        legacy_dates_key: Key name for legacy dates configuration
        legacy_history_days_key: Key name for legacy history days configuration
        logger: Optional logger for migration warnings

    Returns:
        TradingWindowConfig with parsed values
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Initialize with defaults
    trading_window = TradingWindowConfig()

    # Parse new trading_window configuration if present
    if "trading_window" in config:
        tw_config = config["trading_window"]
        if isinstance(tw_config, dict):
            if "start" in tw_config:
                trading_window.start = _parse_date(tw_config["start"])
            if "end" in tw_config:
                trading_window.end = _parse_date(tw_config["end"])
            if "history_backfill_days" in tw_config:
                trading_window.history_backfill_days = int(tw_config["history_backfill_days"])
            if "enforce_trading_window" in tw_config:
                trading_window.enforce_trading_window = bool(tw_config["enforce_trading_window"])

    # Handle legacy dates configuration for backward compatibility
    if legacy_dates_key in config and "dates" not in config:
        dates_config = config[legacy_dates_key]
        if isinstance(dates_config, dict):
            # If trading_window not configured, use legacy dates as trading window
            if trading_window.start is None and "start" in dates_config:
                trading_window.start = _parse_date(dates_config["start"])
            if trading_window.end is None and "end" in dates_config:
                trading_window.end = _parse_date(dates_config["end"])

            # Log migration warning if using legacy format
            logger.warning(
                "Using legacy 'dates' configuration. "
                "Please migrate to 'trading_window' format. "
                "See documentation for new semantics: warmup vs trading window."
            )

    # Handle legacy history days for backward compatibility
    if legacy_history_days_key in config and trading_window.history_backfill_days == 0:
        trading_window.history_backfill_days = int(config[legacy_history_days_key])
        logger.info(
            f"Migrated legacy rolling_history_days={config[legacy_history_days_key]} "
            f"to history_backfill_days={trading_window.history_backfill_days}"
        )

    return trading_window


def compute_load_ranges(
    trading_window: TradingWindowConfig,
) -> Tuple[date, date, date, date]:
    """
    Compute data loading ranges vs trading window ranges.

    Args:
        trading_window: Trading window configuration

    Returns:
        Tuple of (load_start, load_end, trade_start, trade_end)
    """
    if trading_window.start is None or trading_window.end is None:
        raise ValueError("Trading window start and end dates must be configured")

    # Trading window is the configured period
    trade_start = trading_window.start
    trade_end = trading_window.end

    # Load range extends backward for history backfill
    load_start = trade_start - timedelta(days=trading_window.history_backfill_days)
    load_end = trade_end

    return load_start, load_end, trade_start, trade_end


def log_ranges_summary(
    trading_window: TradingWindowConfig,
    logger: logging.Logger,
) -> None:
    """
    Log a one-line summary of load vs trading ranges.

    Args:
        trading_window: Trading window configuration
        logger: Logger instance
    """
    load_start, load_end, trade_start, trade_end = compute_load_ranges(trading_window)

    logger.info(
        f"RANGES load=[{load_start.isoformat()}, {load_end.isoformat()}] "
        f"trade=[{trade_start.isoformat()}, {trade_end.isoformat()}] "
        f"backfill_days={trading_window.history_backfill_days} "
        f"enforce={'ON' if trading_window.enforce_trading_window else 'OFF'}"
    )


def is_date_in_trading_window(
    target_date: date,
    trading_window: TradingWindowConfig,
    enforce: Optional[bool] = None,
) -> bool:
    """
    Check if a date falls within the configured trading window.

    Args:
        target_date: Date to check
        trading_window: Trading window configuration
        enforce: Optional enforcement override (uses config setting if None)

    Returns:
        True if date is within trading window (or enforcement disabled)
    """
    if trading_window.start is None or trading_window.end is None:
        return True  # No trading window configured, allow all dates

    if enforce is None:
        enforce = trading_window.enforce_trading_window

    if not enforce:
        return True  # Enforcement disabled, allow all dates

    return trading_window.start <= target_date <= trading_window.end


def _parse_date(date_input) -> date:
    """Parse date from string or date object."""
    if isinstance(date_input, date):
        return date_input
    elif isinstance(date_input, str):
        return date.fromisoformat(date_input)
    else:
        raise ValueError(f"Cannot parse date from {date_input}: {type(date_input)}")