#!/usr/bin/env python3
"""
Configuration migration helper for trading window feature.

This script helps users migrate from legacy configuration format to the new
trading window format while maintaining backward compatibility.
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import yaml


def detect_config_format(config_data):
    """Detect whether configuration uses legacy or new format."""
    has_legacy_dates = "dates" in config_data
    has_trading_window = "trading_window" in config_data

    if has_trading_window:
        return "new"
    elif has_legacy_dates:
        return "legacy"
    else:
        return "unknown"


def migrate_legacy_config(universe_config, forecast_config, dry_run=False):
    """Migrate legacy configuration to new format."""
    if "dates" not in universe_config:
        print("❌ No legacy dates configuration found")
        return None

    print("🔄 Migrating legacy configuration...")

    # Extract legacy dates
    dates = universe_config["dates"]
    start_date = dates["start"]
    end_date = dates["end"]

    # Get history backfill days from forecast or use default
    history_days = forecast_config.get("rolling_history_days", 90)

    # Create new trading window configuration
    new_trading_window = {
        "start": start_date,
        "end": end_date,
        "history_backfill_days": history_days,
        "enforce_trading_window": False  # Start in permissive mode
    }

    # Create new universe config
    new_universe_config = universe_config.copy()
    new_universe_config["trading_window"] = new_trading_window
    # Keep legacy dates for compatibility during transition

    # Remove rolling_history_days from forecast if present
    new_forecast_config = forecast_config.copy()
    if "rolling_history_days" in new_forecast_config:
        del new_forecast_config["rolling_history_days"]

    if dry_run:
        print("\n📋 Migration Plan:")
        print(f"  Trading window: {start_date} to {end_date}")
        print(f"  History backfill: {history_days} days")
        print(f"  Enforcement: Disabled (permissive mode)")
        print(f"  Legacy 'dates' field: Preserved for compatibility")
        print(f"  Forecast 'rolling_history_days': Moved to trading window")
        return new_universe_config, new_forecast_config
    else:
        print(f"✅ Created trading window: {start_date} to {end_date}")
        print(f"✅ Set history backfill: {history_days} days")
        print(f"✅ Enforcement: Disabled (permissive mode)")
        print(f"✅ Legacy dates preserved for compatibility")
        return new_universe_config, new_forecast_config


def enable_enforcement(universe_config, history_backfill_days=None, dry_run=False):
    """Enable trading window enforcement in existing configuration."""
    if "trading_window" not in universe_config:
        print("❌ No trading window configuration found")
        return None

    print("🔒 Enabling trading window enforcement...")

    new_config = universe_config.copy()
    new_config["trading_window"] = new_config["trading_window"].copy()
    new_config["trading_window"]["enforce_trading_window"] = True

    if history_backfill_days:
        new_config["trading_window"]["history_backfill_days"] = history_backfill_days
        print(f"✅ Updated history backfill: {history_backfill_days} days")

    if dry_run:
        print(f"📋 Will set enforce_trading_window: True")
        if history_backfill_days:
            print(f"📋 Will set history_backfill_days: {history_backfill_days}")
        return new_config
    else:
        print(f"✅ Enabled trading window enforcement")
        return new_config


def validate_config(universe_config, forecast_config):
    """Validate configuration consistency."""
    print("🔍 Validating configuration...")

    issues = []

    # Check universe config
    if "tickers" not in universe_config:
        issues.append("Missing 'tickers' in universe config")

    # Check for mixed legacy/new format
    has_legacy = "dates" in universe_config
    has_new = "trading_window" in universe_config

    if has_legacy and has_new:
        issues.append("Both legacy 'dates' and new 'trading_window' found (new takes priority)")

    # Check trading window if present
    if has_new:
        tw = universe_config["trading_window"]
        if "start" not in tw or "end" not in tw:
            issues.append("Trading window missing 'start' or 'end'")
        elif tw["start"] > tw["end"]:
            issues.append("Trading window start date is after end date")

        if "history_backfill_days" in tw and tw["history_backfill_days"] < 1:
            issues.append("history_backfill_days must be >= 1")

    if issues:
        print("❌ Validation issues found:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("✅ Configuration is valid")
        return True


def generate_migration_summary(universe_config, forecast_config):
    """Generate a summary of the current configuration state."""
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)

    # Detection
    format_type = detect_config_format(universe_config)
    print(f"Configuration format: {format_type}")

    # Universe config info
    tickers = universe_config.get("tickers", [])
    print(f"Symbols: {len(tickers)} ({', '.join(tickers[:3])}{'...' if len(tickers) > 3 else ''})")

    # Trading window info
    if "trading_window" in universe_config:
        tw = universe_config["trading_window"]
        print(f"Trading window: {tw.get('start', 'N/A')} to {tw.get('end', 'N/A')}")
        print(f"History backfill: {tw.get('history_backfill_days', 'N/A')} days")
        print(f"Enforcement: {'Enabled' if tw.get('enforce_trading_window') else 'Disabled'}")
    elif "dates" in universe_config:
        dates = universe_config["dates"]
        print(f"Legacy dates: {dates.get('start', 'N/A')} to {dates.get('end', 'N/A')}")
        print("⚠️  Using legacy format (consider migration)")

    # Forecast config info
    if "rolling_history_days" in forecast_config:
        print(f"Forecast history days: {forecast_config['rolling_history_days']}")

    # Legacy compatibility check
    has_legacy = "dates" in universe_config
    has_new = "trading_window" in universe_config

    if has_legacy and has_new:
        print("⚠️  Mixed configuration (new format takes priority)")
    elif has_legacy:
        print("📋 Legacy format detected (migration recommended)")
    elif has_new:
        print("✅ New trading window format in use")

    # Enforcement status
    if has_new and universe_config["trading_window"].get("enforce_trading_window"):
        print("🔒 Trading window enforcement: ENABLED")
    else:
        print("🔓 Trading window enforcement: DISABLED (permissive mode)")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate TimeGPT v2 configurations to trading window format"
    )
    parser.add_argument(
        "--universe-config",
        required=True,
        help="Path to universe configuration file"
    )
    parser.add_argument(
        "--forecast-config",
        required=True,
        help="Path to forecast configuration file"
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate legacy configuration to new format"
    )
    parser.add_argument(
        "--enable-enforcement",
        action="store_true",
        help="Enable trading window enforcement"
    )
    parser.add_argument(
        "--history-backfill-days",
        type=int,
        help="Set custom history backfill days"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration without migration"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show configuration summary only"
    )

    args = parser.parse_args()

    # Load configurations
    try:
        with open(args.universe_config, 'r') as f:
            universe_config = yaml.safe_load(f)

        with open(args.forecast_config, 'r') as f:
            forecast_config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error loading configuration files: {e}")
        return 1

    # Summary mode
    if args.summary:
        generate_migration_summary(universe_config, forecast_config)
        return 0

    # Validation mode
    if args.validate:
        if validate_config(universe_config, forecast_config):
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return 1
        return 0

    # Migration mode
    if args.migrate:
        if detect_config_format(universe_config) != "legacy":
            print("❌ No legacy configuration to migrate")
            return 1

        new_universe, new_forecast = migrate_legacy_config(
            universe_config, forecast_config, dry_run=args.dry_run
        )

        if not args.dry_run and new_universe and new_forecast:
            # Backup original files
            backup_suffix = ".backup"
            universe_backup = Path(args.universe_config + backup_suffix)
            forecast_backup = Path(args.forecast_config + backup_suffix)

            try:
                Path(args.universe_config).rename(universe_backup)
                Path(args.forecast_config).rename(forecast_backup)
                print(f"✅ Backed up original files")

                # Write new configurations
                with open(args.universe_config, 'w') as f:
                    yaml.dump(new_universe, f, default_flow_style=False)
                with open(args.forecast_config, 'w') as f:
                    yaml.dump(new_forecast, f, default_flow_style=False)

                print(f"✅ Migration completed successfully")
                print(f"📁 Backups: {universe_backup.name}, {forecast_backup.name}")

            except Exception as e:
                print(f"❌ Error writing migrated configuration: {e}")
                return 1

    # Enable enforcement mode
    if args.enable_enforcement:
        if "trading_window" not in universe_config:
            print("❌ No trading window configuration found")
            return 1

        new_config = enable_enforcement(
            universe_config,
            args.history_backfill_days,
            dry_run=args.dry_run
        )

        if not args.dry_run and new_config:
            # Write updated config
            with open(args.universe_config, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            print(f"✅ Trading window enforcement enabled")

    # Always show summary at the end
    generate_migration_summary(universe_config, forecast_config)

    return 0


if __name__ == "__main__":
    sys.exit(main())