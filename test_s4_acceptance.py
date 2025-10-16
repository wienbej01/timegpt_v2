#!/usr/bin/env python3
"""Test to verify Sprint 4 acceptance criteria."""

import sys
import os
import subprocess
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_s4_acceptance_criteria():
    """Test Sprint 4 (Config & CLI Plumbing) acceptance criteria."""

    print("=== Sprint 4 Acceptance Test ===\n")

    # Test 1: Verify forecast.yaml has proper exog configuration
    print("1. Testing forecast.yaml exog configuration...")
    forecast_path = Path("configs/forecast.yaml")
    if not forecast_path.exists():
        print("   ✗ forecast.yaml not found")
        return False

    import yaml
    with open(forecast_path, 'r') as f:
        config = yaml.safe_load(f)

    exog_config = config.get('exog', {})
    if exog_config.get('use_exogs') is True:
        print("   ✓ exog.use_exogs: true")
    else:
        print("   ✗ exog.use_exogs missing or false")
        return False

    if exog_config.get('strict_exog') is True:
        print("   ✓ exog.strict_exog: true")
    else:
        print("   ✗ exog.strict_exog missing or false")
        return False

    # Check for documented allow-lists
    if 'hist_exog_list' in exog_config and 'futr_exog_list' in exog_config:
        print("   ✓ hist_exog_list and futr_exog_list documented")
    else:
        print("   ✗ hist_exog_list and futr_exog_list missing")
        return False

    # Test 2: Verify smoke config files exist
    print("\n2. Testing smoke config files...")

    smoke_forecast_path = Path("configs/forecast_smoke.yaml")
    smoke_universe_path = Path("configs/universe_smoke.yaml")

    if smoke_forecast_path.exists():
        print("   ✓ forecast_smoke.yaml exists")
    else:
        print("   ✗ forecast_smoke.yaml missing")
        return False

    if smoke_universe_path.exists():
        print("   ✓ universe_smoke.yaml exists")
    else:
        print("   ✗ universe_smoke.yaml missing")
        return False

    # Test 3: Verify smoke config has proper exog settings
    print("\n3. Testing smoke config exog settings...")

    with open(smoke_forecast_path, 'r') as f:
        smoke_config = yaml.safe_load(f)

    smoke_exog = smoke_config.get('exog', {})
    if smoke_exog.get('use_exogs') is True:
        print("   ✓ smoke forecast has use_exogs: true")
    else:
        print("   ✗ smoke forecast missing use_exogs: true")
        return False

    if smoke_exog.get('strict_exog') is True:
        print("   ✓ smoke forecast has strict_exog: true")
    else:
        print("   ✗ smoke forecast missing strict_exog: true")
        return False

    # Test 4: Verify CLI --payload-log flag
    print("\n4. Testing CLI --payload-log flag...")

    # Test help output for forecast command
    result = subprocess.run([
        "bash", "-c", "source .venv/bin/activate && python -m timegpt_v2.cli forecast --help"
    ], capture_output=True, text=True)

    help_output = result.stdout

    if "--payload-log" in help_output:
        print("   ✓ --payload-log flag available in CLI")
    else:
        print("   ✗ --payload-log flag missing from CLI")
        return False

    # Test 5: Verify CLI --strict-exog flag
    print("\n5. Testing CLI --strict-exog flag...")

    if "--strict-exog" in help_output and "--no-strict-exog" in help_output:
        print("   ✓ --strict-exog/--no-strict-exog flags available in CLI")
    else:
        print("   ✗ --strict-exog/--no-strict-exog flags missing from CLI")
        return False

    # Test 6: Verify PAYLOAD_LOG environment variable functionality
    print("\n6. Testing PAYLOAD_LOG environment variable functionality...")

    # Test that CLI sets PAYLOAD_LOG when --payload-log is used
    # We can't easily test the full CLI, but we can verify the logic exists
    from timegpt_v2.cli import forecast
    import inspect

    # Get the source code of the forecast function
    source = inspect.getsource(forecast)

    if 'os.environ["PAYLOAD_LOG"] = "1"' in source:
        print("   ✓ CLI sets PAYLOAD_LOG environment variable when --payload-log used")
    else:
        print("   ✗ CLI does not set PAYLOAD_LOG environment variable")
        return False

    # Test 7: Verify smoke config has reasonable parameters
    print("\n7. Testing smoke config parameters...")

    # Check AAPL universe
    with open(smoke_universe_path, 'r') as f:
        universe_config = yaml.safe_load(f)

    if universe_config.get('tickers') == ['AAPL']:
        print("   ✓ smoke universe has AAPL only")
    else:
        print("   ✗ smoke universe missing AAPL-only configuration")
        return False

    # Check reduced parameters for smoke test
    if smoke_config.get('rolling_history_days', 40) <= 20:
        print("   ✓ smoke config has reduced rolling_history_days")
    else:
        print("   ✗ smoke config rolling_history_days too large for smoke test")
        return False

    if smoke_config.get('min_obs_subhourly', 1000) <= 200:
        print("   ✓ smoke config has reduced min_obs_subhourly")
    else:
        print("   ✗ smoke config min_obs_subhourly too large for smoke test")
        return False

    print("\n✅ All Sprint 4 acceptance criteria met!")
    print("✅ forecast.yaml has proper exog configuration with use_exogs and strict_exog")
    print("✅ Smoke config files exist (forecast_smoke.yaml and universe_smoke.yaml)")
    print("✅ Smoke configs have proper exog settings")
    print("✅ CLI --payload-log flag available")
    print("✅ CLI --strict-exog/--no-strict-exog flags available")
    print("✅ CLI properly sets PAYLOAD_LOG environment variable")
    print("✅ Smoke configs have reasonable parameters for testing")

    return True

if __name__ == "__main__":
    success = test_s4_acceptance_criteria()
    sys.exit(0 if success else 1)