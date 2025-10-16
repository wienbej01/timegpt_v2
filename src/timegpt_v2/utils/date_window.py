"""Centralized date window utilities with inclusive end semantics."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Final

import pandas as pd


def get_trading_holidays(years: list[int]) -> list[date]:
    """
    Get a list of NYSE trading holidays for the given years.

    Args:
        years: List of years to get holidays for

    Returns:
        List of holiday dates
    """
    if not years:
        return []

    try:
        import pandas_market_calendars as mcal

        nyse = mcal.get_calendar("NYSE")
        start_date = date(min(years), 1, 1)
        end_date = date(max(years), 12, 31)
        holidays = nyse.holidays().holidays
        return sorted(
            list(
                set(
                    pd.to_datetime(h).date()
                    for h in holidays
                    if start_date <= pd.to_datetime(h).date() <= end_date
                )
            )
        )

    except ImportError:
        # Fallback to a simple list if pandas_market_calendars is not installed
        fallback_holidays: list[date] = []
        for year in years:
            fallback_holidays.extend(
                [
                    date(year, 1, 1),  # New Year's Day
                    date(year, 1, 18),  # Martin Luther King, Jr. Day (observed)
                    date(year, 2, 15),  # Washington's Birthday (observed)
                    date(year, 4, 2),  # Good Friday
                    date(year, 5, 31),  # Memorial Day
                    date(year, 7, 5),  # Independence Day (observed)
                    date(year, 9, 6),  # Labor Day
                    date(year, 11, 25),  # Thanksgiving Day
                    date(year, 12, 24),  # Christmas Day (observed)
                ]
            )
        return sorted(list(set(fallback_holidays)))


def make_trading_day_range(
    start_date: date,
    end_date: date,
    *,
    inclusive_end: bool = True,
    tz: str = "America/New_York",
    holidays: list[date] | None = None,
) -> list[date]:
    """
    Generate a list of trading days between start and end dates.

    This utility centralizes date window logic with consistent inclusive/exclusive
    semantics and proper holiday/weekend handling.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive or exclusive based on inclusive_end flag)
        inclusive_end: Whether to include the end_date in the range (default: True)
        tz: Timezone for date calculations (default: "America/New_York")
        holidays: Optional list of holiday dates to exclude

    Returns:
        List of trading days (weekdays excluding holidays)

    Example:
        >>> make_trading_day_range(
        ...     date(2024, 12, 31),
        ...     date(2025, 1, 2),
        ...     inclusive_end=True
        ... )
        [datetime.date(2024, 12, 31), datetime.date(2025, 1, 2)]
    """
    logger = logging.getLogger(__name__)

    # Adjust end date based on inclusive_end flag
    effective_end = end_date if inclusive_end else end_date - pd.Timedelta(days=1)

    if start_date > effective_end:
        logger.warning(
            f"Date window empty: start_date={start_date} > effective_end={effective_end} "
            f"(original_end={end_date}, inclusive_end={inclusive_end})"
        )
        return []

    # Generate date range
    date_range = pd.date_range(
        start=start_date,
        end=effective_end,
        freq="D",
        tz=tz
    )

    # Convert to date objects
    dates = [d.date() for d in date_range]

    # Filter out weekends
    trading_days = [d for d in dates if d.weekday() < 5]  # Monday=0, Friday=4

    # Filter out holidays
    if holidays is None:
        # Generate holidays for relevant years if not provided
        years = sorted({d.year for d in trading_days})
        holidays = get_trading_holidays(years)

    holiday_set = set(holidays)
    trading_days = [d for d in trading_days if d not in holiday_set]

    logger.info(
        f"DATES start={start_date} end={end_date} inclusive_end={inclusive_end} "
        f"n_days={len(trading_days)} tz={tz}"
    )

    return sorted(trading_days)


def validate_date_window(
    start_date: date,
    end_date: date,
    *,
    inclusive_end: bool = True,
    max_days: int = 365,
) -> None:
    """
    Validate date window parameters and warn about potential issues.

    Args:
        start_date: Start date
        end_date: End date
        inclusive_end: Whether end_date is inclusive
        max_days: Maximum reasonable number of trading days to warn about

    Raises:
        ValueError: If date window is invalid
    """
    logger = logging.getLogger(__name__)

    # Basic validation
    if start_date > end_date:
        raise ValueError(f"start_date {start_date} cannot be after end_date {end_date}")

    # Check for extremely long windows
    effective_end = end_date if inclusive_end else end_date - pd.Timedelta(days=1)
    total_days = (effective_end - start_date).days + 1

    if total_days > max_days * 2:  # Rough estimate including weekends
        logger.warning(
            f"Large date window: {total_days} calendar days between {start_date} and {end_date}. "
            f"This may result in many trading days ({max_days} trading days is typical)."
        )

    # Check for very short windows
    if total_days < 7:
        logger.warning(
            f"Very short date window: only {total_days} calendar days between {start_date} and {end_date}. "
            f"This may result in insufficient trading days for analysis."
        )


def format_date_range_summary(dates: list[date]) -> str:
    """
    Format a concise summary of a date range.

    Args:
        dates: List of dates to summarize

    Returns:
        Formatted summary string
    """
    if not dates:
        return "No dates"

    if len(dates) == 1:
        return f"{dates[0]} (1 day)"

    first_date = dates[0]
    last_date = dates[-1]
    total_days = len(dates)

    # Count gaps (weekends/holidays)
    calendar_days = (last_date - first_date).days + 1
    skip_days = calendar_days - total_days

    return f"{first_date} → {last_date} ({total_days} trading days, {skip_days} skipped)"


# Global constants for easy access
DEFAULT_TZ: Final = "America/New_York"
DEFAULT_INCLUSIVE_END: Final = True

__all__ = [
    "make_trading_day_range",
    "validate_date_window",
    "format_date_range_summary",
    "DEFAULT_TZ",
    "DEFAULT_INCLUSIVE_END",
]