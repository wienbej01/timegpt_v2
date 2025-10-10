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
        *,
        active_windows: Sequence[tuple[time, time]] | None = None,
        max_snapshots_per_day: int | None = None,
        max_total_snapshots: int | None = None,
        skip_dates: Sequence[date] | None = None,
    ) -> None:
        self.dates = sorted(list(set(dates)))
        self.snapshots = sorted(list(set(snapshots)))
        self.tz = ZoneInfo(tz) if isinstance(tz, str) else tz
        self.holidays = set(holidays) if holidays else set()
        self.active_windows = tuple(active_windows) if active_windows else tuple()
        self.max_snapshots_per_day = int(max_snapshots_per_day) if max_snapshots_per_day else None
        self.max_total_snapshots = int(max_total_snapshots) if max_total_snapshots else None
        self.skip_dates = set(skip_dates) if skip_dates else set()

    def _is_trading_day(self, dt: date) -> bool:
        """Check if a date is a trading day (not a weekend or holiday)."""
        if dt in self.holidays:
            return False
        if dt.weekday() >= 5:  # Saturday or Sunday
            return False
        if dt in self.skip_dates:
            return False
        return True

    def generate_snapshots(self) -> list[datetime]:
        """Generate a list of snapshot timestamps."""
        snapshots: list[datetime] = []
        reached_limit = False
        for dt in self.dates:
            if self._is_trading_day(dt):
                day_snapshots: list[datetime] = []
                for snapshot_time in self.snapshots:
                    if not self._within_active_window(snapshot_time):
                        continue
                    day_snapshots.append(datetime.combine(dt, snapshot_time, tzinfo=self.tz))

                if self.max_snapshots_per_day is not None:
                    day_snapshots = day_snapshots[: self.max_snapshots_per_day]

                for snapshot in day_snapshots:
                    snapshots.append(snapshot)
                    if (
                        self.max_total_snapshots is not None
                        and len(snapshots) >= self.max_total_snapshots
                    ):
                        reached_limit = True
                        break
                if reached_limit:
                    break
        return sorted(snapshots)

    def _within_active_window(self, snapshot_time: time) -> bool:
        if not self.active_windows:
            return True
        for start, end in self.active_windows:
            if start <= snapshot_time <= end:
                return True
        return False


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
