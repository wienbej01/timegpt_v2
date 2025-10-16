"""Test snapshot planning across year boundary to ensure inclusive end date behavior."""

from __future__ import annotations

import pytest
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from timegpt_v2.forecast.scheduler import ForecastScheduler


def test_year_boundary_snapshot_planning() -> None:
    """Test that snapshot planning works correctly across year boundary with inclusive end date."""

    # Create a scheduler spanning year boundary
    # 2024-12-31 (Tuesday) to 2025-01-02 (Thursday)
    dates = [
        date(2024, 12, 31),  # Tuesday
        date(2025, 1, 2),    # Thursday (skipping Jan 1 which is Wednesday)
    ]

    # Simple snapshot at 10:00 AM
    snapshots = [time(10, 0)]

    # NY timezone
    tz = ZoneInfo("America/New_York")

    # No holidays to keep it simple
    holidays = []

    scheduler = ForecastScheduler(
        dates=dates,
        snapshots=snapshots,
        tz=tz,
        holidays=holidays,
        active_windows=None,
        max_snapshots_per_day=None,
        max_total_snapshots=None,
        skip_dates=None,
    )

    # Generate snapshots
    planned_snapshots = scheduler.generate_snapshots()

    # Verify we get exactly 2 snapshots (inclusive end date)
    assert len(planned_snapshots) == 2, f"Expected 2 snapshots, got {len(planned_snapshots)}"

    # Verify the dates are correct
    snapshot_dates = [s.date() for s in planned_snapshots]
    assert date(2024, 12, 31) in snapshot_dates, "Missing 2024-12-31 snapshot"
    assert date(2025, 1, 2) in snapshot_dates, "Missing 2025-01-02 snapshot"

    # Verify the times are correct
    for snapshot in planned_snapshots:
        assert snapshot.time() == time(10, 0), f"Wrong snapshot time: {snapshot.time()}"
        assert snapshot.tzinfo == tz, f"Wrong timezone: {snapshot.tzinfo}"

    # Verify chronological order
    assert planned_snapshots[0] < planned_snapshots[1], "Snapshots not in chronological order"


def test_year_boundary_with_holidays() -> None:
    """Test year boundary behavior when there are holidays."""

    # New Year period with Jan 1 as holiday
    dates = [
        date(2024, 12, 30),  # Monday
        date(2024, 12, 31),  # Tuesday
        date(2025, 1, 1),    # Wednesday (holiday)
        date(2025, 1, 2),    # Thursday
    ]

    snapshots = [time(14, 30)]  # 2:30 PM
    tz = ZoneInfo("America/New_York")

    # Mark Jan 1 as holiday
    holidays = [date(2025, 1, 1)]

    scheduler = ForecastScheduler(
        dates=dates,
        snapshots=snapshots,
        tz=tz,
        holidays=holidays,
        active_windows=None,
        max_snapshots_per_day=None,
        max_total_snapshots=None,
        skip_dates=None,
    )

    planned_snapshots = scheduler.generate_snapshots()

    # Should get 3 snapshots (Jan 1 skipped due to holiday)
    assert len(planned_snapshots) == 3, f"Expected 3 snapshots, got {len(planned_snapshots)}"

    snapshot_dates = [s.date() for s in planned_snapshots]
    assert date(2024, 12, 30) in snapshot_dates, "Missing 2024-12-30"
    assert date(2024, 12, 31) in snapshot_dates, "Missing 2024-12-31"
    assert date(2025, 1, 2) in snapshot_dates, "Missing 2025-01-02"
    assert date(2025, 1, 1) not in snapshot_dates, "Jan 1 should be skipped (holiday)"


def test_weekend_handling_across_year_boundary() -> None:
    """Test that weekends are properly filtered across year boundary."""

    # 2024-12-29 (Sunday) to 2025-01-04 (Saturday)
    dates = [
        date(2024, 12, 29),  # Sunday (weekend)
        date(2024, 12, 30),  # Monday
        date(2024, 12, 31),  # Tuesday
        date(2025, 1, 1),    # Wednesday
        date(2025, 1, 2),    # Thursday
        date(2025, 1, 3),    # Friday
        date(2025, 1, 4),    # Saturday (weekend)
    ]

    snapshots = [time(9, 30)]
    tz = ZoneInfo("America/New_York")
    holidays = []  # No holidays for this test

    scheduler = ForecastScheduler(
        dates=dates,
        snapshots=snapshots,
        tz=tz,
        holidays=holidays,
        active_windows=None,
        max_snapshots_per_day=None,
        max_total_snapshots=None,
        skip_dates=None,
    )

    planned_snapshots = scheduler.generate_snapshots()

    # Should get 5 snapshots (excluding weekends)
    assert len(planned_snapshots) == 5, f"Expected 5 snapshots, got {len(planned_snapshots)}"

    snapshot_dates = [s.date() for s in planned_snapshots]

    # Verify weekends are excluded
    assert date(2024, 12, 29) not in snapshot_dates, "Sunday should be excluded"
    assert date(2025, 1, 4) not in snapshot_dates, "Saturday should be excluded"

    # Verify weekdays are included
    assert date(2024, 12, 30) in snapshot_dates, "Monday should be included"
    assert date(2024, 12, 31) in snapshot_dates, "Tuesday should be included"
    assert date(2025, 1, 1) in snapshot_dates, "Wednesday should be included"
    assert date(2025, 1, 2) in snapshot_dates, "Thursday should be included"
    assert date(2025, 1, 3) in snapshot_dates, "Friday should be included"


def test_multiple_snapshots_per_day_across_year_boundary() -> None:
    """Test multiple snapshot times per day across year boundary."""

    dates = [
        date(2024, 12, 31),
        date(2025, 1, 1),
        date(2025, 1, 2),
    ]

    # Multiple snapshots per day
    snapshots = [time(10, 0), time(14, 0), time(16, 30)]
    tz = ZoneInfo("America/New_York")
    holidays = []

    scheduler = ForecastScheduler(
        dates=dates,
        snapshots=snapshots,
        tz=tz,
        holidays=holidays,
        active_windows=None,
        max_snapshots_per_day=None,
        max_total_snapshots=None,
        skip_dates=None,
    )

    planned_snapshots = scheduler.generate_snapshots()

    # Should get 9 snapshots (3 days × 3 snapshots per day)
    assert len(planned_snapshots) == 9, f"Expected 9 snapshots, got {len(planned_snapshots)}"

    # Group by date and verify each day has 3 snapshots
    from collections import defaultdict
    daily_groups = defaultdict(list)
    for snapshot in planned_snapshots:
        daily_groups[snapshot.date()].append(snapshot)

    for test_date in dates:
        assert len(daily_groups[test_date]) == 3, f"Expected 3 snapshots for {test_date}, got {len(daily_groups[test_date])}"

        # Verify snapshot times for each day
        times = sorted([s.time() for s in daily_groups[test_date]])
        expected_times = [time(10, 0), time(14, 0), time(16, 30)]
        assert times == expected_times, f"Wrong snapshot times for {test_date}: {times}"


def test_year_boundary_with_new_constructor() -> None:
    """Test year boundary using the new date range constructor."""

    # Test year boundary with inclusive end
    start_date = date(2024, 12, 31)  # Tuesday
    end_date = date(2025, 1, 2)      # Thursday

    snapshots = [time(10, 0)]
    tz = ZoneInfo("America/New_York")

    scheduler = ForecastScheduler.create_with_date_range(
        start_date=start_date,
        end_date=end_date,
        snapshots=snapshots,
        tz=tz,
        inclusive_end=True,
    )

    planned_snapshots = scheduler.generate_snapshots()

    # Should get 2 snapshots (Tuesday and Thursday, skip Wednesday New Year's)
    assert len(planned_snapshots) == 2, f"Expected 2 snapshots, got {len(planned_snapshots)}"

    snapshot_dates = [s.date() for s in planned_snapshots]
    assert date(2024, 12, 31) in snapshot_dates, "Missing 2024-12-31 snapshot"
    assert date(2025, 1, 2) in snapshot_dates, "Missing 2025-01-02 snapshot"


def test_year_boundary_exclusive_with_new_constructor() -> None:
    """Test year boundary with exclusive end using new constructor."""

    start_date = date(2024, 12, 31)  # Tuesday
    end_date = date(2025, 1, 2)      # Thursday

    snapshots = [time(10, 0)]
    tz = ZoneInfo("America/New_York")

    scheduler = ForecastScheduler.create_with_date_range(
        start_date=start_date,
        end_date=end_date,
        snapshots=snapshots,
        tz=tz,
        inclusive_end=False,  # Exclusive end
    )

    planned_snapshots = scheduler.generate_snapshots()

    # Should get only 1 snapshot (Tuesday, Thursday excluded due to exclusive end)
    assert len(planned_snapshots) == 1, f"Expected 1 snapshot, got {len(planned_snapshots)}"
    assert planned_snapshots[0].date() == date(2024, 12, 31)


def test_active_windows_filtering_across_year_boundary() -> None:
    """Test that active windows properly filter snapshots across year boundary."""

    dates = [
        date(2024, 12, 31),
        date(2025, 1, 1),
        date(2025, 1, 2),
    ]

    # Snapshots throughout the day
    snapshots = [time(9, 0), time(10, 30), time(14, 0), time(16, 0)]
    tz = ZoneInfo("America/New_York")
    holidays = []

    # Active window: 10:00 to 15:00
    active_windows = [(time(10, 0), time(15, 0))]

    scheduler = ForecastScheduler(
        dates=dates,
        snapshots=snapshots,
        tz=tz,
        holidays=holidays,
        active_windows=active_windows,
        max_snapshots_per_day=None,
        max_total_snapshots=None,
        skip_dates=None,
    )

    planned_snapshots = scheduler.generate_snapshots()

    # Should get 6 snapshots (3 days × 2 snapshots within active window)
    assert len(planned_snapshots) == 6, f"Expected 6 snapshots, got {len(planned_snapshots)}"

    # Verify only snapshots within active window are included
    for snapshot in planned_snapshots:
        snapshot_time = snapshot.time()
        assert time(10, 0) <= snapshot_time <= time(15, 0), f"Snapshot {snapshot_time} outside active window"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])