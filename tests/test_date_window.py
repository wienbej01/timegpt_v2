"""Test date window utilities with inclusive/exclusive semantics."""

from __future__ import annotations

import pytest
from datetime import date, timedelta

from timegpt_v2.utils.date_window import (
    make_trading_day_range,
    validate_date_window,
    format_date_range_summary,
    get_trading_holidays,
)


class TestTradingDayRange:
    """Test make_trading_day_range function."""

    def test_basic_inclusive_end(self):
        """Test basic functionality with inclusive end."""
        start = date(2024, 12, 30)  # Monday
        end = date(2025, 1, 2)      # Thursday

        dates = make_trading_day_range(start, end, inclusive_end=True)

        # Should include Monday, Tuesday, Thursday (Wednesday is New Year's holiday)
        expected = [
            date(2024, 12, 30),  # Monday
            date(2024, 12, 31),  # Tuesday
            date(2025, 1, 2),    # Thursday
        ]
        assert dates == expected

    def test_exclusive_end(self):
        """Test exclusive end semantics."""
        start = date(2024, 12, 30)  # Monday
        end = date(2025, 1, 2)      # Thursday

        dates = make_trading_day_range(start, end, inclusive_end=False)

        # Should not include Thursday (end date is exclusive)
        expected = [
            date(2024, 12, 30),  # Monday
            date(2024, 12, 31),  # Tuesday
        ]
        assert dates == expected

    def test_year_boundary_inclusive(self):
        """Test year boundary with inclusive end."""
        # Test case from sprint plan: 2024-12-31 to 2025-01-02
        start = date(2024, 12, 31)  # Tuesday
        end = date(2025, 1, 2)      # Thursday

        dates = make_trading_day_range(start, end, inclusive_end=True)

        # Should include Tuesday and Thursday (skip Wednesday New Year's)
        expected = [
            date(2024, 12, 31),  # Tuesday
            date(2025, 1, 2),    # Thursday
        ]
        assert dates == expected
        assert len(dates) == 2

    def test_year_boundary_exclusive(self):
        """Test year boundary with exclusive end."""
        start = date(2024, 12, 31)  # Tuesday
        end = date(2025, 1, 2)      # Thursday

        dates = make_trading_day_range(start, end, inclusive_end=False)

        # Should only include Tuesday (end date exclusive)
        expected = [
            date(2024, 12, 31),  # Tuesday
        ]
        assert dates == expected
        assert len(dates) == 1

    def test_weekend_filtering(self):
        """Test that weekends are properly filtered."""
        # Range that includes weekend
        start = date(2024, 12, 27)  # Friday
        end = date(2024, 12, 31)    # Tuesday

        dates = make_trading_day_range(start, end, inclusive_end=True)

        # Should skip Saturday and Sunday
        expected = [
            date(2024, 12, 27),  # Friday
            date(2024, 12, 30),  # Monday
            date(2024, 12, 31),  # Tuesday
        ]
        assert dates == expected

    def test_custom_holidays(self):
        """Test with custom holiday list."""
        start = date(2024, 12, 30)  # Monday
        end = date(2025, 1, 3)      # Friday

        # Make Tuesday a custom holiday (Note: when holidays are provided,
        # they completely replace the default holidays, so we need to include New Year's too)
        custom_holidays = [date(2024, 12, 31), date(2025, 1, 1)]  # Tuesday + New Year's

        dates = make_trading_day_range(start, end, inclusive_end=True, holidays=custom_holidays)

        # Should skip Tuesday (custom holiday) and Wednesday (New Year's custom holiday)
        expected = [
            date(2024, 12, 30),  # Monday
            date(2025, 1, 2),    # Thursday
            date(2025, 1, 3),    # Friday
        ]
        assert dates == expected

    def test_single_day_range(self):
        """Test range with single day."""
        start = date(2024, 12, 30)  # Monday
        end = date(2024, 12, 30)    # Monday

        dates = make_trading_day_range(start, end, inclusive_end=True)
        assert dates == [date(2024, 12, 30)]

        dates = make_trading_day_range(start, end, inclusive_end=False)
        assert dates == []  # Exclusive end means no days

    def test_empty_range(self):
        """Test with invalid range (start > end)."""
        start = date(2025, 1, 2)    # Thursday
        end = date(2024, 12, 30)    # Monday

        dates = make_trading_day_range(start, end, inclusive_end=True)
        assert dates == []

    def test_weekend_only_range(self):
        """Test range that only contains weekends."""
        start = date(2024, 12, 28)  # Saturday
        end = date(2024, 12, 29)    # Sunday

        dates = make_trading_day_range(start, end, inclusive_end=True)
        assert dates == []  # No trading days

    def test_multiple_weeks(self):
        """Test range spanning multiple weeks."""
        start = date(2024, 12, 23)  # Monday
        end = date(2025, 1, 6)      # Monday

        dates = make_trading_day_range(start, end, inclusive_end=True)

        # Should include all weekdays except holidays and weekends
        # Verify start and end are included
        assert dates[0] == date(2024, 12, 23)
        assert dates[-1] == date(2025, 1, 6)

        # Should not include weekends
        for d in dates:
            assert d.weekday() < 5  # Monday-Friday only

        # Should be sorted
        assert dates == sorted(dates)


class TestDateWindowValidation:
    """Test validate_date_window function."""

    def test_valid_range(self):
        """Test with valid date range."""
        start = date(2024, 12, 30)
        end = date(2025, 1, 2)

        # Should not raise
        validate_date_window(start, end)

    def test_invalid_range(self):
        """Test with invalid date range."""
        start = date(2025, 1, 2)
        end = date(2024, 12, 30)

        with pytest.raises(ValueError, match="start_date .* cannot be after end_date"):
            validate_date_window(start, end)

    def test_large_range_warning(self):
        """Test warning for large date range."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 1) + timedelta(days=800)  # Very large range

        # Should log warning but not raise
        validate_date_window(start, end, max_days=365)

    def test_short_range_warning(self):
        """Test warning for very short range."""
        start = date(2024, 12, 30)
        end = date(2024, 12, 31)

        # Should log warning but not raise
        validate_date_window(start, end)


class TestDateRangeSummary:
    """Test format_date_range_summary function."""

    def test_empty_list(self):
        """Test with empty date list."""
        summary = format_date_range_summary([])
        assert summary == "No dates"

    def test_single_date(self):
        """Test with single date."""
        dates = [date(2024, 12, 30)]
        summary = format_date_range_summary(dates)
        assert summary == "2024-12-30 (1 day)"

    def test_multiple_dates(self):
        """Test with multiple dates."""
        dates = [
            date(2024, 12, 30),  # Monday
            date(2024, 12, 31),  # Tuesday
            date(2025, 1, 2),    # Thursday (skip Wednesday holiday)
        ]
        summary = format_date_range_summary(dates)
        assert "2024-12-30 → 2025-01-02" in summary
        assert "3 trading days" in summary
        assert "1 skipped" in summary

    def test_consecutive_days(self):
        """Test with consecutive trading days."""
        dates = [
            date(2024, 12, 30),  # Monday
            date(2024, 12, 31),  # Tuesday
        ]
        summary = format_date_range_summary(dates)
        assert "2 trading days" in summary
        assert "0 skipped" in summary


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_timezone_handling(self):
        """Test that timezone parameter is accepted but doesn't break functionality."""
        start = date(2024, 12, 30)
        end = date(2025, 1, 2)

        # Should work with different timezone - note that holidays are timezone-aware
        # In UTC, New Year's might be treated differently, but we still expect weekend filtering
        dates = make_trading_day_range(start, end, tz="UTC")

        # Should include Monday, Tuesday, Thursday (skip weekend, New Year's handled by holiday logic)
        expected_dates = [date(2024, 12, 30), date(2024, 12, 31), date(2025, 1, 2)]
        assert dates == expected_dates
        assert len(dates) == 3

    def test_leap_year_handling(self):
        """Test date range including leap year."""
        start = date(2024, 2, 28)  # Wednesday
        end = date(2024, 3, 1)     # Friday

        dates = make_trading_day_range(start, end, inclusive_end=True)

        # Should include Feb 28, Feb 29 (leap day), and Mar 1
        expected = [
            date(2024, 2, 28),  # Wednesday
            date(2024, 2, 29),  # Thursday (leap day)
            date(2024, 3, 1),    # Friday
        ]
        assert dates == expected

    def test_cross_month_boundary(self):
        """Test range crossing month boundary."""
        start = date(2024, 12, 31)  # Tuesday
        end = date(2025, 1, 31)     # Friday

        dates = make_trading_day_range(start, end, inclusive_end=True)

        # Should include dates from both months
        assert dates[0] == date(2024, 12, 31)
        assert dates[-1] == date(2025, 1, 31)
        assert len(dates) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])