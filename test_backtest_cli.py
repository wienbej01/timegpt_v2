#!/usr/bin/env python3
"""
Focused CLI Backtest Parameter Test Script
Tests backtest command with CLI parameters using existing successful run data
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import shutil

def run_command(cmd: str, description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        print(f"EXIT CODE: {result.returncode}")
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")

        return result.returncode == 0, result.stdout + result.stderr

    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out after 5 minutes")
        return False, "Command timed out"
    except Exception as e:
        print(f"ERROR: Failed to run command: {e}")
        return False, str(e)

def check_artifacts(run_id: str) -> dict[str, bool]:
    """Check if expected artifacts exist."""
    base_path = Path(f"artifacts/runs/{run_id}")
    artifacts = {
        "validation_data": (base_path / "validation" / "clean.parquet").exists(),
        "features": (base_path / "features" / "features.parquet").exists(),
        "forecasts": (base_path / "forecasts" / "quantiles.csv").exists(),
        "trades": (base_path / "trades" / "bt_trades.csv").exists(),
        "eval_summary": (base_path / "eval" / "bt_summary.csv").exists(),
        "portfolio_metrics": (base_path / "eval" / "portfolio_summary.csv").exists(),
        "per_symbol_metrics": (base_path / "eval" / "per_symbol_summary.csv").exists(),
    }

    print(f"\nARTIFACTS CHECK:")
    for artifact_name, exists in artifacts.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {artifact_name}")

    return artifacts

def analyze_trades(run_id: str) -> dict:
    """Analyze generated trades if they exist."""
    trades_file = Path(f"artifacts/runs/{run_id}/trades/bt_trades.csv")

    if not trades_file.exists():
        return {"error": "No trades file found"}

    try:
        # Simple analysis without pandas to avoid dependency issues
        with open(trades_file, 'r') as f:
            lines = f.readlines()

        if len(lines) <= 1:  # Only header
            return {"error": "No trades generated", "lines": len(lines)}

        # Parse CSV manually (basic)
        trades = []
        headers = lines[0].strip().split(',')

        for line in lines[1:]:
            values = line.strip().split(',')
            if len(values) == len(headers):
                trade = dict(zip(headers, values))
                trades.append(trade)

        # Basic statistics
        symbols = set(trade.get('symbol', 'UNKNOWN') for trade in trades)
        total_pnl = sum(float(trade.get('pnl', 0)) for trade in trades if trade.get('pnl'))
        winning_trades = len([t for t in trades if float(t.get('pnl', 0)) > 0])
        total_trades = len(trades)

        return {
            "total_trades": total_trades,
            "symbols_traded": list(symbols),
            "total_pnl": total_pnl,
            "winning_trades": winning_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
            "sample_trades": trades[:3]  # First 3 trades for inspection
        }

    except Exception as e:
        return {"error": f"Failed to analyze trades: {e}"}

def setup_test_environment(source_run_id: str, target_run_id: str):
    """Copy existing successful run data to test CLI parameter overrides."""
    source_path = Path(f"artifacts/runs/{source_run_id}")
    target_path = Path(f"artifacts/runs/{target_run_id}")

    if source_path.exists() and not target_path.exists():
        print(f"Copying existing data from {source_run_id} to {target_run_id}")
        shutil.copytree(source_path, target_path)
        return True
    elif target_path.exists():
        print(f"Target run {target_run_id} already exists")
        return True
    else:
        print(f"Source run {source_run_id} not found")
        return False

def main():
    """Main test function focused on CLI parameter overrides for backtest."""
    print("FOCUSED CLI BACKTEST PARAMETER TEST")
    print("=" * 80)
    print("Testing backtest command with CLI parameter overrides using existing data")

    # Use existing successful run as base
    source_run_id = "tsla_febmar2025_1760595513"
    target_run_id = f"cli_backtest_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nTEST CONFIGURATION:")
    print(f"  SOURCE RUN ID: {source_run_id}")
    print(f"  TARGET RUN ID: {target_run_id}")
    print(f"  TESTING: CLI parameter overrides for tickers and dates")

    # Setup test environment
    if not setup_test_environment(source_run_id, target_run_id):
        print("❌ Failed to setup test environment")
        return False

    # Check initial artifacts
    print(f"\nINITIAL ARTIFACTS:")
    initial_artifacts = check_artifacts(target_run_id)

    if not initial_artifacts.get("forecasts", False):
        print("❌ No forecasts found in source data - cannot test backtest")
        return False

    print("✅ Found required artifacts for backtest testing")

    # Test 1: Backtest with original parameters (should work)
    original_cmd = (
        f'python -m timegpt_v2.cli backtest '
        f'--config-dir configs --config-name forecast.yaml '
        f'--universe-name universe_tsla_febmar2025.yaml '
        f'--run-id "{target_run_id}"'
    )

    print(f"\n{'='*80}")
    print("TEST 1: BACKTEST WITH ORIGINAL CONFIGURATION")
    print("=" * 80)

    success1, output1 = run_command(original_cmd, "Backtest with original config")
    artifacts1 = check_artifacts(target_run_id)
    trades1 = analyze_trades(target_run_id) if artifacts1.get("trades", False) else {"error": "No trades"}

    # Test 2: Backtest with CLI parameter overrides
    # Use TSLA only and a subset of dates that are within the original range
    override_cmd = (
        f'python -m timegpt_v2.cli backtest '
        f'--config-dir configs --config-name forecast.yaml '
        f'--universe-name universe_tsla_febmar2025.yaml '
        f'--tickers "TSLA" '
        f'--start-date "2025-02-01" '
        f'--end-date "2025-02-28" '
        f'--run-id "{target_run_id}"'
    )

    print(f"\n{'='*80}")
    print("TEST 2: BACKTEST WITH CLI PARAMETER OVERRIDES")
    print("=" * 80)

    success2, output2 = run_command(override_cmd, "Backtest with CLI parameter overrides")
    artifacts2 = check_artifacts(target_run_id)
    trades2 = analyze_trades(target_run_id) if artifacts2.get("trades", False) else {"error": "No trades"}

    # Analysis
    print(f"\n{'='*80}")
    print("TEST RESULTS ANALYSIS")
    print("=" * 80)

    print(f"\nTEST 1 - Original Configuration:")
    print(f"  Success: {'✅' if success1 else '❌'}")
    print(f"  Trades Generated: {'✅' if artifacts1.get('trades', False) else '❌'}")
    print(f"  Performance Summary: {'✅' if artifacts1.get('eval_summary', False) else '❌'}")
    if trades1.get("total_trades"):
        print(f"  Total Trades: {trades1['total_trades']}")
        print(f"  Win Rate: {trades1['win_rate']:.2%}")

    print(f"\nTEST 2 - CLI Parameter Overrides:")
    print(f"  Success: {'✅' if success2 else '❌'}")
    print(f"  Trades Generated: {'✅' if artifacts2.get('trades', False) else '❌'}")
    print(f"  Performance Summary: {'✅' if artifacts2.get('eval_summary', False) else '❌'}")
    if trades2.get("total_trades"):
        print(f"  Total Trades: {trades2['total_trades']}")
        print(f"  Win Rate: {trades2['win_rate']:.2%}")

    # Final verdict
    overall_success = success1 and success2 and artifacts1.get("trades", False) and artifacts2.get("trades", False)

    print(f"\n{'='*80}")
    print("FINAL VERDICT")
    print("=" * 80)

    if overall_success:
        print("✅ CLI PARAMETER TEST PASSED")
        print("  Both original configuration and CLI parameter overrides work correctly")
        print("  Backtest generates trades and performance reports with parameter overrides")
    else:
        print("❌ CLI PARAMETER TEST FAILED")
        if not success1:
            print("  Original configuration failed - base issue")
        if not success2:
            print("  CLI parameter overrides failed - parameter override issue")
        if not artifacts1.get("trades", False) or not artifacts2.get("trades", False):
            print("  No trades generated - trading logic issue")

    print(f"\n📁 Test Results: artifacts/runs/{target_run_id}/")
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)