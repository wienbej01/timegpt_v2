"""Tests for Sprint 5: Snapshot cadence aligned to horizon."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from timegpt_v2.forecast.scheduler import (
    create_horizon_preset,
    create_snapshot_preset,
    ForecastScheduler,
    iter_snapshots_with_coverage,
)
from timegpt_v2.utils.coverage import CoverageTracker, SkipReason


class TestHorizonPresets:
    """Test horizon-aligned snapshot presets."""

    def test_30m_standard_preset(self):
        """Test 30-minute standard preset configuration."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date
        )

        # Check basic properties
        assert isinstance(scheduler, ForecastScheduler)
        assert len(scheduler.snapshots) > 0
        assert scheduler.max_snapshots_per_day > 0

        # Generate snapshots and verify cadence
        snapshots = scheduler.generate_snapshots()
        assert len(snapshots) > 0

        # All snapshots should be within RTH (9:30-16:00 ET)
        rth_open = time(9, 30)
        rth_close = time(16, 0)
        for snap in snapshots:
            assert rth_open <= snap.time() < rth_close

        # Should exclude first and last 10 minutes
        first_snap = snapshots[0].time()
        assert first_snap >= time(9, 40)  # At least 10 minutes after open

        last_snap = snapshots[-1].time()
        assert last_snap < time(15, 50)  # At least 10 minutes before close

    def test_60m_standard_preset(self):
        """Test 60-minute standard preset configuration."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        scheduler = create_horizon_preset(
            horizon_minutes=60,
            start_date=start_date,
            end_date=end_date
        )

        # Generate snapshots and verify cadence
        snapshots = scheduler.generate_snapshots()
        assert len(snapshots) > 0

        # Should have fewer snapshots than 30m preset
        snapshot_times = [snap.time() for snap in snapshots]
        expected_times = [time(10, 30), time(12, 0), time(14, 30)]
        assert all(time in snapshot_times for time in expected_times)

        # Check spacing between snapshots within the same day
        # Get unique times only (ignore duplicates across days)
        unique_times = sorted(set(snapshot_times))
        if len(unique_times) > 1:
            for i in range(1, len(unique_times)):
                prev_minutes = (unique_times[i-1].hour - 9) * 60 + unique_times[i-1].minute - 30
                curr_minutes = (unique_times[i].hour - 9) * 60 + unique_times[i].minute - 30
                # Should be at least 30 minutes apart for 60m horizon
                assert (curr_minutes - prev_minutes) >= 30

    def test_high_frequency_preset(self):
        """Test high frequency preset configuration."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date,
            exclude_first_minutes=5,
            exclude_last_minutes=5,
            max_trades_per_day=20
        )

        # Should allow more trades per day
        assert scheduler.max_snapshots_per_day == 20

        # Should exclude fewer minutes at boundaries
        snapshots = scheduler.generate_snapshots()
        first_snap = snapshots[0].time()
        assert first_snap >= time(9, 35)  # Only 5 minutes after open

    def test_invalid_horizon_minutes(self):
        """Test that invalid horizon minutes raise error."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        with pytest.raises(ValueError, match="horizon_minutes must be 30 or 60"):
            create_horizon_preset(
                horizon_minutes=45,  # Invalid horizon
                start_date=start_date,
                end_date=end_date
            )

    def test_trading_window_filter(self):
        """Test trading window filtering functionality."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)
        trading_start = date(2024, 1, 3)
        trading_end = date(2024, 1, 7)

        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date,
            trading_window_start=trading_start,
            trading_window_end=trading_end
        )

        snapshots = scheduler.generate_snapshots()
        snapshot_dates = set(snap.date() for snap in snapshots)

        # Should only include trading window dates
        assert all(trading_start <= date <= trading_end for date in snapshot_dates)
        assert snapshot_dates.issubset({trading_start, trading_end, date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 6)})

    def test_max_trades_per_day_enforcement(self):
        """Test max trades per day enforcement."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 3)

        # Custom max trades per day
        max_trades = 5
        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date,
            max_trades_per_day=max_trades
        )

        # Should limit snapshots per day
        assert scheduler.max_snapshots_per_day == max_trades

        # Count snapshots per trading day
        snapshots = scheduler.generate_snapshots()
        daily_counts = {}
        for snap in snapshots:
            day = snap.date()
            daily_counts[day] = daily_counts.get(day, 0) + 1

        # Each day should have <= max_trades snapshots
        for day, count in daily_counts.items():
            assert count <= max_trades


class TestSnapshotPresets:
    """Test predefined snapshot presets."""

    def test_30m_standard_preset(self):
        """Test 30m standard predefined preset."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        scheduler = create_snapshot_preset(
            preset_name="30m_standard",
            start_date=start_date,
            end_date=end_date
        )

        snapshots = scheduler.generate_snapshots()
        assert len(snapshots) > 0

        # Should be same as horizon_minutes=30 preset
        direct_scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date
        )
        direct_snapshots = direct_scheduler.generate_snapshots()

        assert len(snapshots) == len(direct_snapshots)

    def test_60m_standard_preset(self):
        """Test 60m standard predefined preset."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        scheduler = create_snapshot_preset(
            preset_name="60m_standard",
            start_date=start_date,
            end_date=end_date
        )

        snapshots = scheduler.generate_snapshots()
        assert len(snapshots) > 0

        # Should be same as horizon_minutes=60 preset
        direct_scheduler = create_horizon_preset(
            horizon_minutes=60,
            start_date=start_date,
            end_date=end_date
        )
        direct_snapshots = direct_scheduler.generate_snapshots()

        assert len(snapshots) == len(direct_snapshots)

    def test_high_frequency_preset(self):
        """Test high frequency predefined preset."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 3)

        scheduler = create_snapshot_preset(
            preset_name="high_frequency",
            start_date=start_date,
            end_date=end_date
        )

        # Should allow 20 trades per day
        assert scheduler.max_snapshots_per_day == 20

        # Should exclude fewer boundary minutes
        snapshots = scheduler.generate_snapshots()
        first_snap = snapshots[0].time()
        assert first_snap >= time(9, 35)  # Only 5 minutes after open

    def test_invalid_preset_name(self):
        """Test that invalid preset names raise error."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        with pytest.raises(ValueError, match="Unknown preset"):
            create_snapshot_preset(
                preset_name="invalid_preset",
                start_date=start_date,
                end_date=end_date
            )

    def test_preset_with_custom_kwargs(self):
        """Test that custom kwargs are passed through."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        scheduler = create_snapshot_preset(
            preset_name="30m_standard",
            start_date=start_date,
            end_date=end_date,
            max_trades_per_day=8,  # Custom parameter
            exclude_first_minutes=15  # Custom parameter
        )

        assert scheduler.max_snapshots_per_day == 8

        # Should respect custom exclude minutes
        snapshots = scheduler.generate_snapshots()
        first_snap = snapshots[0].time()
        assert first_snap >= time(9, 45)  # 15 minutes after open


class TestCoverageTracking:
    """Test coverage tracking with horizon-aligned presets."""

    def test_coverage_tracking_30m(self):
        """Test coverage tracking with 30m preset."""
        # Create test features
        timestamps = pd.date_range("2024-01-01 09:30", periods=500, freq="1min", tz="America/New_York")
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 500,
            "timestamp": timestamps,
            "target_log_return_1m": np.random.normal(0, 0.02, 500),
            "is_rth": True
        })

        # Create 30m scheduler (use Jan 2nd since Jan 1st is holiday)
        scheduler = create_snapshot_preset(
            preset_name="30m_standard",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2)
        )

        coverage_tracker = CoverageTracker()
        logger = pd.DataFrame()  # Mock logger

        # Generate snapshots with coverage tracking
        snapshots_list = list(iter_snapshots_with_coverage(
            features=features,
            scheduler=scheduler,
            min_obs_subhourly=50,  # Lower threshold for test
            logger=logger,
            coverage_tracker=coverage_tracker
        ))

        # Should have planned and sent snapshots
        coverage_summary = coverage_tracker.get_coverage_summary()
        assert coverage_summary["counters"]["planned"] > 0
        assert coverage_summary["counters"]["sent"] > 0
        assert coverage_summary["counters"]["sent"] <= coverage_summary["counters"]["planned"]

        # Should have coverage metrics
        assert "coverage_pct" in coverage_summary
        assert coverage_summary["coverage_pct"] > 0

    def test_coverage_tracking_with_skip_reasons(self):
        """Test coverage tracking records skip reasons."""
        # Create minimal features that will likely fail warm-up gate
        timestamps = pd.date_range("2024-01-01 09:30", periods=20, freq="1min", tz="America/New_York")
        features = pd.DataFrame({
            "symbol": ["AAPL"] * 20,
            "timestamp": timestamps,
            "target_log_return_1m": np.random.normal(0, 0.02, 20),
            "is_rth": True
        })

        # Create scheduler with snapshots after feature data
        scheduler = create_snapshot_preset(
            preset_name="30m_standard",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 2)
        )

        coverage_tracker = CoverageTracker()
        logger = pd.DataFrame()  # Mock logger

        # Generate snapshots with coverage tracking
        snapshots_list = list(iter_snapshots_with_coverage(
            features=features,
            scheduler=scheduler,
            min_obs_subhourly=100,  # High threshold to trigger skips
            logger=logger,
            coverage_tracker=coverage_tracker
        ))

        # Should have skipped snapshots due to insufficient history
        coverage_summary = coverage_tracker.get_coverage_summary()
        assert coverage_summary["counters"]["skipped"] > 0

        # Check skip reasons
        skip_reasons = coverage_summary["skip_reasons"]
        assert skip_reasons.get(SkipReason.GATE_MIN_OBS, 0) > 0


class TestTurnoverControl:
    """Test turnover control mechanisms."""

    def test_turnover_limits_30m(self):
        """Test turnover limits for 30m horizon."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 3)  # 3 trading days

        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date,
            max_trades_per_day=8  # Low limit
        )

        snapshots = scheduler.generate_snapshots()
        daily_counts = {}
        for snap in snapshots:
            day = snap.date()
            daily_counts[day] = daily_counts.get(day, 0) + 1

        # Each day should have exactly 8 snapshots (max limit)
        for day, count in daily_counts.items():
            assert count == 8

    def test_turnover_limits_60m(self):
        """Test turnover limits for 60m horizon."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)  # 5 trading days

        scheduler = create_horizon_preset(
            horizon_minutes=60,
            start_date=start_date,
            end_date=end_date,
            max_trades_per_day=6  # Even lower limit for longer horizon
        )

        snapshots = scheduler.generate_snapshots()
        daily_counts = {}
        for snap in snapshots:
            day = snap.date()
            daily_counts[day] = daily_counts.get(day, 0) + 1

        # Each day should have at most 6 snapshots
        for day, count in daily_counts.items():
            assert count <= 6

    def test_total_snapshot_limit(self):
        """Test total snapshot limit enforcement."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 10)  # 10 trading days

        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date,
            max_total_snapshots=15  # Very low total limit
        )

        snapshots = scheduler.generate_snapshots()
        assert len(snapshots) <= 15


class TestRTHAlignment:
    """Test RTH (Regular Trading Hours) alignment."""

    def test_snapshots_within_rth(self):
        """Test that all snapshots are within RTH."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date
        )

        snapshots = scheduler.generate_snapshots()

        rth_open = time(9, 30)
        rth_close = time(16, 0)

        # All snapshots should be within RTH
        for snap in snapshots:
            snap_time = snap.time()
            assert rth_open <= snap_time < rth_close, f"Snapshot {snap_time} outside RTH [{rth_open}, {rth_close})"

    def test_weekend_exclusion(self):
        """Test that weekends are excluded."""
        # Include a weekend in the date range
        start_date = date(2024, 1, 1)  # Monday
        end_date = date(2024, 1, 7)   # Sunday

        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date
        )

        snapshots = scheduler.generate_snapshots()

        # Should only have weekday snapshots
        weekday_count = 0
        for snap in snapshots:
            if snap.weekday() < 5:  # Monday=0, Friday=4
                weekday_count += 1

        assert weekday_count > 0
        assert weekday_count == len(snapshots)  # All should be weekdays

    def test_snapshot_spacing_30m(self):
        """Test appropriate spacing for 30m horizon."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 3)

        scheduler = create_horizon_preset(
            horizon_minutes=30,
            start_date=start_date,
            end_date=end_date
        )

        snapshots = scheduler.generate_snapshots()

        # Extract times for a single day
        day_snapshots = [s for s in snapshots if s.date() == date(2024, 1, 1)]
        day_times = sorted(s.time() for s in day_snapshots)

        # Check spacing between consecutive snapshots
        if len(day_times) > 1:
            for i in range(1, len(day_times)):
                prev_minutes = (day_times[i-1].hour - 9) * 60 + day_times[i-1].minute - 30
                curr_minutes = (day_times[i].hour - 9) * 60 + day_times[i].minute - 30

                # Should be exactly 30 minutes apart for 30m horizon
                assert curr_minutes - prev_minutes == 30

    def test_snapshot_spacing_60m(self):
        """Test appropriate spacing for 60m horizon."""
        start_date = date(2024, 1, 1)
        end_date = date(2024, 1, 5)

        scheduler = create_horizon_preset(
            horizon_minutes=60,
            start_date=start_date,
            end_date=end_date
        )

        snapshots = scheduler.generate_snapshots()

        # Extract times for a single trading day (Jan 1 is holiday)
        day_snapshots = [s for s in snapshots if s.date() == date(2024, 1, 2)]
        day_times = sorted(s.time() for s in day_snapshots)

        # Should have standard 60m preset times
        expected_times = [time(10, 30), time(12, 0), time(14, 30)]
        assert day_times == expected_times


if __name__ == "__main__":
    pytest.main([__file__, "-v"])