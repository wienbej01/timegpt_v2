from __future__ import annotations

from datetime import date, time

import pandas as pd
import pytest

from timegpt_v2.forecast.scheduler import ForecastScheduler, get_trading_holidays


@pytest.fixture
def sample_dates() -> list[date]:
    """Return a sequence of dates for testing."""
    return list(pd.to_datetime(pd.date_range(start="2024-07-01", end="2024-07-10")).date)


@pytest.fixture
def sample_snapshots() -> list[time]:
    """Return a sequence of snapshot times for testing."""
    return [time(10, 0), time(14, 30)]


@pytest.fixture
def sample_holidays() -> list[date]:
    """Return a sequence of holidays for testing."""
    return [date(2024, 7, 4)]  # Independence Day


def test_scheduler_generates_correct_snapshot_count(
    sample_dates: list[date],
    sample_snapshots: list[time],
    sample_holidays: list[date],
) -> None:
    """Test that the scheduler generates the correct number of snapshots."""
    scheduler = ForecastScheduler(
        dates=sample_dates,
        snapshots=sample_snapshots,
        tz="America/New_York",
        holidays=sample_holidays,
    )
    snapshots = scheduler.generate_snapshots()

    # 10 days total
    # - 2 weekend days (July 6, 7)
    # - 1 holiday (July 4)
    # = 7 trading days
    # 7 trading days * 2 snapshots/day = 14 snapshots
    assert len(snapshots) == 14


def test_scheduler_skips_weekends_and_holidays(
    sample_dates: list[date],
    sample_snapshots: list[time],
    sample_holidays: list[date],
) -> None:
    """Test that the scheduler skips weekends and holidays."""
    scheduler = ForecastScheduler(
        dates=sample_dates,
        snapshots=sample_snapshots,
        tz="America/New_York",
        holidays=sample_holidays,
    )
    snapshots = scheduler.generate_snapshots()
    snapshot_dates = {s.date() for s in snapshots}

    assert date(2024, 7, 4) not in snapshot_dates
    assert date(2024, 7, 6) not in snapshot_dates
    assert date(2024, 7, 7) not in snapshot_dates


def test_scheduler_is_deterministic(
    sample_dates: list[date],
    sample_snapshots: list[time],
    sample_holidays: list[date],
) -> None:
    """Test that the scheduler produces a deterministic list of snapshots."""
    scheduler1 = ForecastScheduler(
        dates=sample_dates,
        snapshots=sample_snapshots,
        tz="America/New_York",
        holidays=sample_holidays,
    )
    snapshots1 = scheduler1.generate_snapshots()

    scheduler2 = ForecastScheduler(
        dates=sample_dates,
        snapshots=sample_snapshots,
        tz="America/New_York",
        holidays=sample_holidays,
    )
    snapshots2 = scheduler2.generate_snapshots()

    assert snapshots1 == snapshots2


def test_get_trading_holidays() -> None:
    """Test that trading holidays can be retrieved."""
    holidays = get_trading_holidays(years=[2024])
    assert date(2024, 1, 1) in holidays
    assert date(2024, 7, 4) in holidays
    assert date(2024, 12, 25) in holidays


def test_scheduler_active_windows_and_quota() -> None:
    """Scheduler should respect active windows and snapshot quotas."""
    dates = [date(2024, 7, 1), date(2024, 7, 2)]
    snapshots = [time(9, 30), time(10, 0), time(15, 0), time(16, 0)]
    scheduler = ForecastScheduler(
        dates=dates,
        snapshots=snapshots,
        tz="America/New_York",
        holidays=[],
        active_windows=[(time(9, 45), time(15, 30))],
        max_snapshots_per_day=2,
        max_total_snapshots=3,
    )
    generated = scheduler.generate_snapshots()

    assert len(generated) == 3
    assert all(time(9, 45) <= snap.time() <= time(15, 30) for snap in generated)
    assert generated[0].date() == date(2024, 7, 1)
    assert generated[1].date() == date(2024, 7, 1)
    assert generated[2].date() == date(2024, 7, 2)
