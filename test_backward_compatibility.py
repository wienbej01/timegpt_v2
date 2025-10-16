#!/usr/bin/env python3
"""
Backward compatibility test for trading window feature.

This script tests that existing configurations continue to work as expected
while validating the new trading window functionality.
"""

import json
import logging
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

from timegpt_v2.config.model import TradingWindowConfig
from timegpt_v2.utils.trading_window import parse_trading_window_config, log_ranges_summary


def test_legacy_config_migration():
    """Test that legacy configurations are properly migrated."""
    print("Testing legacy configuration migration...")

    # Simulate legacy configuration
    legacy_config = {
        "dates": {
            "start": "2024-01-01",
            "end": "2024-03-31"
        },
        "tickers": ["AAPL", "MSFT"]
    }

    # Simulate forecast config
    forecast_config = {
        "rolling_history_days": 90
    }

    # Parse with backward compatibility
    logger = logging.getLogger("test")
    trading_window = parse_trading_window_config(
        {**legacy_config, **forecast_config},
        logger=logger
    )

    # Verify migration
    assert trading_window.start == date(2024, 1, 1), f"Expected start=2024-01-01, got {trading_window.start}"
    assert trading_window.end == date(2024, 3, 31), f"Expected end=2024-03-31, got {trading_window.end}"
    assert trading_window.history_backfill_days == 90, f"Expected history_backfill_days=90, got {trading_window.history_backfill_days}"
    assert trading_window.enforce_trading_window == False, f"Expected enforce_trading_window=False, got {trading_window.enforce_trading_window}"

    print("✅ Legacy configuration migration works correctly")
    return trading_window


def test_new_config_format():
    """Test new trading window configuration format."""
    print("Testing new configuration format...")

    # New configuration format
    new_config = {
        "trading_window": {
            "start": "2024-02-01",
            "end": "2024-04-30",
            "history_backfill_days": 120,
            "enforce_trading_window": True
        },
        "tickers": ["AAPL", "MSFT"]
    }

    forecast_config = {"rolling_history_days": 90}  # Should be overridden

    logger = logging.getLogger("test")
    trading_window = parse_trading_window_config(
        {**new_config, **forecast_config},
        logger=logger
    )

    # Verify new format
    assert trading_window.start == date(2024, 2, 1), f"Expected start=2024-02-01, got {trading_window.start}"
    assert trading_window.end == date(2024, 4, 30), f"Expected end=2024-04-30, got {trading_window.end}"
    assert trading_window.history_backfill_days == 120, f"Expected history_backfill_days=120, got {trading_window.history_backfill_days}"
    assert trading_window.enforce_trading_window == True, f"Expected enforce_trading_window=True, got {trading_window.enforce_trading_window}"

    print("✅ New configuration format works correctly")
    return trading_window


def test_mixed_config_priority():
    """Test that new configuration takes priority over legacy."""
    print("Testing mixed configuration priority...")

    # Mixed configuration (both legacy and new present)
    mixed_config = {
        "dates": {
            "start": "2024-01-01",  # Legacy - should be ignored
            "end": "2024-03-31"      # Legacy - should be ignored
        },
        "trading_window": {
            "start": "2024-02-01",  # New - should take priority
            "end": "2024-04-30",      # New - should take priority
            "history_backfill_days": 150,
            "enforce_trading_window": True
        },
        "tickers": ["AAPL", "MSFT"]
    }

    forecast_config = {"rolling_history_days": 90}

    logger = logging.getLogger("test")
    trading_window = parse_trading_window_config(
        {**mixed_config, **forecast_config},
        logger=logger
    )

    # Verify new config takes priority
    assert trading_window.start == date(2024, 2, 1), f"Expected start=2024-02-01 (new), got {trading_window.start}"
    assert trading_window.end == date(2024, 4, 30), f"Expected end=2024-04-30 (new), got {trading_window.end}"
    assert trading_window.history_backfill_days == 150, f"Expected history_backfill_days=150 (new), got {trading_window.history_backfill_days}"
    assert trading_window.enforce_trading_window == True, f"Expected enforce_trading_window=True (new), got {trading_window.enforce_trading_window}"

    print("✅ Mixed configuration priority works correctly")
    return trading_window


def test_default_values():
    """Test that missing fields get sensible defaults."""
    print("Testing default values...")

    # Minimal configuration
    minimal_config = {
        "tickers": ["AAPL"]
    }

    forecast_config = {}  # No forecast config

    logger = logging.getLogger("test")
    trading_window = parse_trading_window_config(
        {**minimal_config, **forecast_config},
        logger=logger
    )

    # Verify defaults
    assert trading_window.start is None, f"Expected start=None, got {trading_window.start}"
    assert trading_window.end is None, f"Expected end=None, got {trading_window.end}"
    assert trading_window.history_backfill_days == 90, f"Expected history_backfill_days=90 (default), got {trading_window.history_backfill_days}"
    assert trading_window.enforce_trading_window == False, f"Expected enforce_trading_window=False (default), got {trading_window.enforce_trading_window}"

    print("✅ Default values work correctly")
    return trading_window


def test_config_serialization():
    """Test that TradingWindowConfig can be serialized/deserialized."""
    print("Testing configuration serialization...")

    original = TradingWindowConfig(
        start=date(2024, 1, 15),
        end=date(2024, 6, 30),
        history_backfill_days=180,
        enforce_trading_window=True
    )

    # Convert to dict and back
    config_dict = original.to_dict()
    restored = TradingWindowConfig.from_dict(config_dict)

    # Verify restoration
    assert restored.start == original.start, f"Start mismatch: {restored.start} != {original.start}"
    assert restored.end == original.end, f"End mismatch: {restored.end} != {original.end}"
    assert restored.history_backfill_days == original.history_backfill_days, f"History backfill mismatch: {restored.history_backfill_days} != {original.history_backfill_days}"
    assert restored.enforce_trading_window == original.enforce_trading_window, f"Enforcement mismatch: {restored.enforce_trading_window} != {original.enforce_trading_window}"

    print("✅ Configuration serialization works correctly")
    return original


def test_config_file_compatibility():
    """Test loading configurations from YAML files."""
    print("Testing configuration file compatibility...")

    # Create temporary YAML files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        legacy_yaml = {
            'dates': {
                'start': '2024-01-01',
                'end': '2024-12-31'
            },
            'tickers': ['AAPL', 'MSFT']
        }
        yaml.dump(legacy_yaml, f)
        legacy_file = Path(f.name)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        new_yaml = {
            'trading_window': {
                'start': '2024-01-01',
                'end': '2024-12-31',
                'history_backfill_days': 120,
                'enforce_trading_window': True
            },
            'tickers': ['AAPL', 'MSFT']
        }
        yaml.dump(new_yaml, f)
        new_file = Path(f.name)

    try:
        # Test legacy file
        with legacy_file.open('r') as f:
            legacy_data = yaml.safe_load(f)

        forecast_config = {'rolling_history_days': 90}
        logger = logging.getLogger("test")
        legacy_parsed = parse_trading_window_config(
            {**legacy_data, **forecast_config},
            logger=logger
        )

        assert legacy_parsed.start == date(2024, 1, 1)
        assert legacy_parsed.enforce_trading_window == False  # Default for legacy

        # Test new file
        with new_file.open('r') as f:
            new_data = yaml.safe_load(f)

        new_parsed = parse_trading_window_config(
            {**new_data, **forecast_config},
            logger=logger
        )

        assert new_parsed.start == date(2024, 1, 1)
        assert new_parsed.history_backfill_days == 120
        assert new_parsed.enforce_trading_window == True

        print("✅ Configuration file compatibility works correctly")

    finally:
        # Cleanup
        legacy_file.unlink()
        new_file.unlink()


def run_all_tests():
    """Run all backward compatibility tests."""
    print("=" * 60)
    print("BACKWARD COMPATIBILITY TEST SUITE")
    print("=" * 60)

    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

    tests = [
        test_legacy_config_migration,
        test_new_config_format,
        test_mixed_config_priority,
        test_default_values,
        test_config_serialization,
        test_config_file_compatibility,
    ]

    results = []
    for test_func in tests:
        try:
            test_func()
            results.append((test_func.__name__, "PASS", None))
        except Exception as e:
            results.append((test_func.__name__, "FAIL", str(e)))
            print(f"❌ {test_func.__name__} failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)

    for test_name, status, error in results:
        status_icon = "✅" if status == "PASS" else "❌"
        print(f"{status_icon} {test_name}: {status}")
        if error:
            print(f"   Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All backward compatibility tests passed!")
        print("The trading window feature maintains full backward compatibility.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review migration compatibility.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)