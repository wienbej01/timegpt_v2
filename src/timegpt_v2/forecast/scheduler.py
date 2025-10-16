from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from timegpt_v2.utils.coverage import CoverageTracker, SkipReason
from timegpt_v2.utils.date_window import make_trading_day_range, validate_date_window, get_trading_holidays


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
        self.max_snapshots_per_day = int(max_snapshots_per_day) if max_snapshots_per_day is not None else None
        self.max_total_snapshots = int(max_total_snapshots) if max_total_snapshots is not None else None
        self.skip_dates = set(skip_dates) if skip_dates else set()

    @classmethod
    def create_with_date_range(
        cls,
        start_date: date,
        end_date: date,
        snapshots: Sequence[time],
        tz: str | ZoneInfo,
        holidays: Sequence[date] | None = None,
        *,
        inclusive_end: bool = True,
        active_windows: Sequence[tuple[time, time]] | None = None,
        max_snapshots_per_day: int | None = None,
        max_total_snapshots: int | None = None,
        skip_dates: Sequence[date] | None = None,
    ) -> ForecastScheduler:
        """
        Create a ForecastScheduler using centralized date window utilities.

        This constructor uses the new date_window utility to generate trading days
        with consistent inclusive/exclusive semantics and proper validation.

        Args:
            start_date: Start date for trading day generation
            end_date: End date for trading day generation
            snapshots: Daily snapshot times
            tz: Timezone for timestamps
            holidays: Optional holiday list (will generate if not provided)
            inclusive_end: Whether end_date is inclusive (default: True)
            active_windows: Active trading hour windows
            max_snapshots_per_day: Maximum snapshots per day
            max_total_snapshots: Maximum total snapshots
            skip_dates: Additional dates to skip

        Returns:
            Configured ForecastScheduler instance
        """
        # Validate the date window
        validate_date_window(start_date, end_date, inclusive_end=inclusive_end)

        # Generate trading days using centralized utility
        trading_days = make_trading_day_range(
            start_date=start_date,
            end_date=end_date,
            inclusive_end=inclusive_end,
            tz=str(tz) if isinstance(tz, ZoneInfo) else tz,
            holidays=holidays,
        )

        # Convert skip_dates to set for proper handling
        skip_dates_set = set(skip_dates) if skip_dates else set()

        # Filter trading days by skip_dates
        filtered_days = [d for d in trading_days if d not in skip_dates_set]

        # Create the scheduler
        return cls(
            dates=filtered_days,
            snapshots=snapshots,
            tz=tz,
            holidays=holidays,
            active_windows=active_windows,
            max_snapshots_per_day=max_snapshots_per_day,
            max_total_snapshots=max_total_snapshots,
            skip_dates=skip_dates,
        )

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


def iter_snapshots(
    features: pd.DataFrame,
    scheduler: ForecastScheduler,
    min_obs_subhourly: int,
    logger: logging.Logger,
) -> Iterable[tuple[str, datetime]]:
    """
    Iterate through snapshots and symbols, applying a warm-up gate.

    This function yields (symbol, snapshot_ts) tuples for combinations that have
    enough historical data.
    """
    all_symbols = sorted(features["symbol"].unique())
    for snapshot_ts in scheduler.generate_snapshots():
        for symbol in all_symbols:
            prior_rows = features[
                (features["timestamp"] < snapshot_ts)
                & (features["symbol"] == symbol)
                & (features["is_rth"])
            ]
            if len(prior_rows) < min_obs_subhourly:
                logger.info(
                    f"SKIP snapshot {snapshot_ts} {symbol} --- history {len(prior_rows)} < {min_obs_subhourly}"
                )
                continue
            yield symbol, snapshot_ts


def iter_snapshots_with_coverage(
    features: pd.DataFrame,
    scheduler: ForecastScheduler,
    min_obs_subhourly: int,
    logger: logging.Logger,
    coverage_tracker: CoverageTracker,
    trading_window=None,
) -> Iterable[tuple[str, datetime]]:
    """
    Iterate through snapshots and symbols with detailed coverage tracking.

    This function yields (symbol, snapshot_ts) tuples for combinations that have
    enough historical data, while tracking all skip reasons and coverage metrics.
    """
    all_symbols = sorted(features["symbol"].unique())

    # First, plan all snapshots
    all_snapshot_timestamps = scheduler.generate_snapshots()

    # Track planned snapshots
    for snapshot_ts in all_snapshot_timestamps:
        for symbol in all_symbols:
            coverage_tracker.add_planned(f"{snapshot_ts.isoformat()}_{symbol}")

    # Now iterate and check warm-up gate
    for snapshot_ts in all_snapshot_timestamps:
        snapshot_date = snapshot_ts.date()

        # Check trading window enforcement if provided
        if trading_window is not None:
            from timegpt_v2.utils.trading_window import is_date_in_trading_window
            if not is_date_in_trading_window(snapshot_date, trading_window):
                # Skip all symbols for this snapshot due to trading window
                for symbol in all_symbols:
                    if snapshot_date < trading_window.start:
                        coverage_tracker.add_skipped(
                            f"{snapshot_ts.isoformat()}_{symbol}",
                            SkipReason.SKIP_BEFORE_TRADE_WINDOW,
                            f"Date {snapshot_date} before trading window start {trading_window.start}"
                        )
                    else:
                        coverage_tracker.add_skipped(
                            f"{snapshot_ts.isoformat()}_{symbol}",
                            SkipReason.SKIP_AFTER_TRADE_WINDOW,
                            f"Date {snapshot_date} after trading window end {trading_window.end}"
                        )
                continue

        # Check if snapshot is within active window
        snapshot_time = snapshot_ts.time()
        if not scheduler._within_active_window(snapshot_time):
            # Skip all symbols for this snapshot due to RTH window
            for symbol in all_symbols:
                coverage_tracker.add_skipped(
                    f"{snapshot_ts.isoformat()}_{symbol}",
                    SkipReason.GATE_RTH,
                    f"Snapshot time {snapshot_time} outside active window"
                )
            continue

        for symbol in all_symbols:
            prior_rows = features[
                (features["timestamp"] < snapshot_ts)
                & (features["symbol"] == symbol)
                & (features["is_rth"])
            ]
            if len(prior_rows) < min_obs_subhourly:
                coverage_tracker.add_skipped(
                    f"{snapshot_ts.isoformat()}_{symbol}",
                    SkipReason.GATE_MIN_OBS,
                    f"History {len(prior_rows)} < {min_obs_subhourly}"
                )
                logger.info(
                    f"SKIP snapshot {snapshot_ts} {symbol} --- history {len(prior_rows)} < {min_obs_subhourly}"
                )
                continue

            # This snapshot passes all gates
            coverage_tracker.add_sent(f"{snapshot_ts.isoformat()}_{symbol}")
            yield symbol, snapshot_ts


def create_horizon_preset(
    horizon_minutes: int,
    *,
    start_date: date,
    end_date: date,
    tz: str = "America/New_York",
    holidays: Sequence[date] | None = None,
    exclude_first_minutes: int = 10,
    exclude_last_minutes: int = 10,
    max_trades_per_day: int | None = None,
    max_total_snapshots: int | None = None,
    trading_window_start: date | None = None,
    trading_window_end: date | None = None,
) -> ForecastScheduler:
    """Create a scheduler with horizon-aligned snapshot presets.

    This function creates snapshot schedules optimized for specific forecast horizons,
    with appropriate cadence and turnover limits.

    Args:
        horizon_minutes: Forecast horizon (30 or 60 minutes)
        start_date: Start date for trading days
        end_date: End date for trading days
        tz: Timezone for timestamps
        holidays: Optional holiday list
        exclude_first_minutes: Minutes to exclude at market open (avoid noise)
        exclude_last_minutes: Minutes to exclude at market close
        max_trades_per_day: Maximum trades per day (for turnover control)
        max_total_snapshots: Maximum total snapshots across all days
        trading_window_start: Optional trading window start date
        trading_window_end: Optional trading window end date

    Returns:
        Configured ForecastScheduler with horizon-aligned snapshots
    """
    if horizon_minutes not in [30, 60]:
        raise ValueError(f"horizon_minutes must be 30 or 60, got {horizon_minutes}")

    # Define RTH active window (9:30 AM - 4:00 PM ET)
    rth_open = time(9, 30)
    rth_close = time(16, 0)

    # Calculate snapshot intervals based on horizon
    if horizon_minutes == 30:
        # Every 30 minutes within RTH, excluding first/last few minutes
        snapshot_times = []
        current_minutes = exclude_first_minutes
        while current_minutes + horizon_minutes <= (390 - exclude_last_minutes):
            # Create datetime at market open and add minutes
            market_open = time(9, 30)
            total_minutes = 9 * 60 + 30 + current_minutes  # 9:30 AM + current_minutes
            snapshot_hour = total_minutes // 60
            snapshot_minute = total_minutes % 60
            snapshot_times.append(time(snapshot_hour, snapshot_minute))
            current_minutes += 30
    else:  # horizon_minutes == 60
        # Every 60 minutes within RTH
        snapshot_times = [
            time(10, 30),  # 10:30 AM (1 hour after open)
            time(12, 0),  # 12:00 PM (lunch)
            time(14, 30),  # 2:30 PM (afternoon)
        ]
        # Filter out snapshots that would end after close
        filtered_times = []
        for snap_time in snapshot_times:
            snap_minutes = ((snap_time.hour - 9) * 60 + snap_time.minute - 30)
            if snap_minutes + horizon_minutes <= (390 - exclude_last_minutes):
                filtered_times.append(snap_time)
        snapshot_times = filtered_times

    # Set default max trades per day based on horizon
    if max_trades_per_day is None:
        max_trades_per_day = 12 if horizon_minutes == 30 else 8

    # Apply trading window if specified
    if trading_window_start and trading_window_end:
        # Filter dates to trading window
        if start_date < trading_window_start:
            start_date = trading_window_start
        if end_date > trading_window_end:
            end_date = trading_window_end

    # Create active window (RTH only)
    active_windows = [(rth_open, rth_close)]

    # Set default max total snapshots if not provided
    if max_total_snapshots is None:
        max_total_snapshots = max_trades_per_day * 100  # Reasonable upper limit

    return ForecastScheduler.create_with_date_range(
        start_date=start_date,
        end_date=end_date,
        snapshots=snapshot_times,
        tz=tz,
        holidays=holidays,
        active_windows=active_windows,
        max_snapshots_per_day=max_trades_per_day,
        max_total_snapshots=max_total_snapshots,
    )


def create_snapshot_preset(
    preset_name: str,
    *,
    start_date: date,
    end_date: date,
    tz: str = "America/New_York",
    holidays: Sequence[date] | None = None,
    **kwargs
) -> ForecastScheduler:
    """Create a scheduler from a predefined preset.

    Args:
        preset_name: Name of the preset ("30m_standard", "60m_standard", "high_frequency")
        start_date: Start date for trading days
        end_date: End date for trading days
        tz: Timezone for timestamps
        holidays: Optional holiday list
        **kwargs: Additional parameters to pass to the preset function

    Returns:
        Configured ForecastScheduler
    """
    presets = {
        "30m_standard": lambda: create_horizon_preset(
            horizon_minutes=30, start_date=start_date, end_date=end_date, tz=tz, holidays=holidays, **kwargs
        ),
        "60m_standard": lambda: create_horizon_preset(
            horizon_minutes=60, start_date=start_date, end_date=end_date, tz=tz, holidays=holidays, **kwargs
        ),
        "high_frequency": lambda: create_horizon_preset(
            horizon_minutes=30, start_date=start_date, end_date=end_date, tz=tz, holidays=holidays,
            exclude_first_minutes=5, exclude_last_minutes=5, max_trades_per_day=20, **kwargs
        ),
    }

    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

    return presets[preset_name]()


# get_trading_holidays is now imported from timegpt_v2.utils.date_window
