from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd


class ForecastScheduler:
    """Generates a schedule of forecast snapshot timestamps."""

    def __init__(
        self,
        dates: Sequence[date],
        snapshots: Sequence[time],
        tz: str | ZoneInfo,
        holidays: Sequence[date] | None = None,
    ) -> None:
        self.dates = sorted(list(set(dates)))
        self.snapshots = sorted(list(set(snapshots)))
        self.tz = ZoneInfo(tz) if isinstance(tz, str) else tz
        self.holidays = set(holidays) if holidays else set()

    def _is_trading_day(self, dt: date) -> bool:
        """Check if a date is a trading day (not a weekend or holiday)."""
        if dt in self.holidays:
            return False
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        return True

    def generate_snapshots(self) -> list[datetime]:
        """Generate a list of snapshot timestamps."""
        snapshots: list[datetime] = []
        for dt in self.dates:
            if self._is_trading_day(dt):
                for t in self.snapshots:
                    snapshots.append(datetime.combine(dt, t, tzinfo=self.tz))
        return sorted(snapshots)


def get_trading_holidays(years: Sequence[int]) -> list[date]:
    """
    Get a list of NYSE trading holidays for the given years.
    """
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
