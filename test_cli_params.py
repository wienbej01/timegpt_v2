#!/usr/bin/env python3
"""
Comprehensive CLI Parameter Test Script
Tests the complete pipeline flow with CLI parameter overrides for tickers and dates
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

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
            timeout=300  # 5 minute timeout per step
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

def check_artifacts(run_id: str, step: str) -> dict[str, bool]:
    """Check if expected artifacts exist for a given step."""
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

    print(f"\nARTIFACTS CHECK for {step}:")
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

def main():
    """Main test function."""
    print("COMPREHENSIVE CLI PARAMETER TEST")
    print("=" * 80)
    print("Testing ticker and date parameter overrides through the complete pipeline")

    # Test configuration
    tickers = "TSLA,AAPL,MSFT"
    start_date = "2025-02-01"
    end_date = "2025-02-28"  # Shorter period for faster testing
    run_id = f"cli_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\nTEST CONFIGURATION:")
    print(f"  TICKERS: {tickers}")
    print(f"  START DATE: {start_date}")
    print(f"  END DATE: {end_date}")
    print(f"  RUN ID: {run_id}")

    # Pipeline commands
    commands = [
        (
            f'python -m timegpt_v2.cli check-data '
            f'--config-dir configs --config-name forecast.yaml '
            f'--universe-name universe.yaml '
            f'--tickers "{tickers}" --start-date "{start_date}" --end-date "{end_date}" '
            f'--run-id "{run_id}"',
            "Data Validation with CLI Parameter Overrides"
        ),
        (
            f'python -m timegpt_v2.cli build-features '
            f'--config-dir configs --config-name forecast.yaml '
            f'--universe-name universe.yaml '
            f'--tickers "{tickers}" --start-date "{start_date}" --end-date "{end_date}" '
            f'--run-id "{run_id}"',
            "Feature Engineering with CLI Parameter Overrides"
        ),
        (
            f'python -m timegpt_v2.cli forecast '
            f'--config-dir configs --config-name forecast.yaml '
            f'--run-id "{run_id}" '
            f'--api-mode offline',
            "Forecast Generation (Offline Mode)"
        ),
        (
            f'python -m timegpt_v2.cli backtest '
            f'--config-dir configs --config-name forecast.yaml '
            f'--universe-name universe.yaml '
            f'--tickers "{tickers}" --start-date "{start_date}" --end-date "{end_date}" '
            f'--run-id "{run_id}"',
            "Backtest with CLI Parameter Overrides - KEY TEST"
        ),
        (
            f'python -m timegpt_v2.cli evaluate '
            f'--config-dir configs '
            f'--run-id "{run_id}"',
            "Performance Evaluation"
        )
    ]

    # Execute pipeline
    results = {}
    for i, (cmd, description) in enumerate(commands, 1):
        step_name = description.split()[0].lower()
        success, output = run_command(cmd, description)
        results[step_name] = {"success": success, "output": output}

        # Check artifacts after each step
        artifacts = check_artifacts(run_id, description)
        results[step_name]["artifacts"] = artifacts

        if not success:
            print(f"\n❌ STEP {i} FAILED: {description}")
            print("Stopping pipeline test")
            break
        else:
            print(f"\n✅ STEP {i} COMPLETED: {description}")

    # Final analysis
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS")
    print(f"{'='*80}")

    # Check final state
    final_artifacts = check_artifacts(run_id, "FINAL STATE")

    # Analyze trades if they exist
    trade_analysis = analyze_trades(run_id)
    print(f"\nTRADE ANALYSIS:")
    print(json.dumps(trade_analysis, indent=2))

    # Summary
    print(f"\nPIPELINE SUMMARY:")
    successful_steps = sum(1 for step in results.values() if step["success"])
    total_steps = len(results)
    print(f"  Successful Steps: {successful_steps}/{total_steps}")

    if final_artifacts.get("trades", False) and final_artifacts.get("eval_summary", False):
        print("  ✅ CLI PARAMETER TEST PASSED: Trades and performance reports generated")
        print(f"  📊 Run ID: {run_id}")
        print(f"  📁 Results location: artifacts/runs/{run_id}/")
    else:
        print("  ❌ CLI PARAMETER TEST FAILED: Missing trades or performance reports")
        print("  🔍 Debugging information:")
        for step_name, step_result in results.items():
            if not step_result["success"]:
                print(f"    - {step_name}: Failed")
            missing_artifacts = [name for name, exists in step_result.get("artifacts", {}).items() if not exists]
            if missing_artifacts:
                print(f"    - {step_name}: Missing artifacts: {missing_artifacts}")

    return successful_steps == total_steps and final_artifacts.get("trades", False)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)