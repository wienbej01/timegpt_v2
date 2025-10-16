#!/usr/bin/env python3
"""
Smoke test script for TimeGPT v2 E2E pipeline.
Runs a minimal end-to-end test to ensure the system produces at least one trade.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd: list[str]) -> bool:
    """Run a command and return True if successful."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
        if result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Exception running command {' '.join(cmd)}: {e}")
        return False

def main():
    run_id = os.environ.get("RUN_ID", "smoke_aapl_2024")
    print(f"Running smoke test with RUN_ID={run_id}")

    commands = [
        [
            "python", "-m", "timegpt_v2.cli", "check-data",
            "--config-name", "forecast_smoke.yaml",
            "--universe-name", "universe_smoke.yaml",
            "--run-id", run_id
        ],
        [
            "python", "-m", "timegpt_v2.cli", "build-features",
            "--config-name", "forecast_smoke.yaml",
            "--universe-name", "universe_smoke.yaml",
            "--run-id", run_id
        ],
        [
            "python", "-m", "timegpt_v2.cli", "forecast",
            "--config-name", "forecast_smoke.yaml",
            "--universe-name", "universe_smoke.yaml",
            "--run-id", run_id
        ],
        [
            "python", "-m", "timegpt_v2.cli", "backtest",
            "--config-name", "forecast_smoke.yaml",
            "--run-id", run_id
        ]
    ]

    for cmd in commands:
        if not run_command(cmd):
            print("❌ Smoke test failed")
            sys.exit(1)

    # Check success criteria
    run_dir = Path("artifacts") / "runs" / run_id

    # Check trades CSV exists
    trades_path = run_dir / "trades" / "bt_trades.csv"
    if not trades_path.exists():
        print(f"❌ Trades CSV not found: {trades_path}")
        sys.exit(1)
    print(f"✅ Trades CSV exists: {trades_path}")

    # Check portfolio/eval CSVs exist
    portfolio_path = run_dir / "eval" / "portfolio_summary.csv"
    if not portfolio_path.exists():
        print(f"❌ Portfolio summary CSV not found: {portfolio_path}")
        sys.exit(1)
    print(f"✅ Portfolio summary CSV exists: {portfolio_path}")

    eval_path = run_dir / "eval" / "bt_summary.csv"
    if not eval_path.exists():
        print(f"❌ Eval summary CSV not found: {eval_path}")
        sys.exit(1)
    print(f"✅ Eval summary CSV exists: {eval_path}")

    # Check trade_count > 0 in eval/bt_summary.csv
    import pandas as pd
    try:
        summary_df = pd.read_csv(eval_path)
        trade_count = summary_df.iloc[0]["trade_count"]
        if trade_count <= 0:
            print(f"❌ trade_count not > 0: {trade_count}")
            sys.exit(1)
        print(f"✅ trade_count > 0: {trade_count}")
    except Exception as e:
        print(f"❌ Error reading eval summary: {e}")
        sys.exit(1)

    # Check funnel line present in logs/backtest.log
    log_path = run_dir / "logs" / "backtest.log"
    if not log_path.exists():
        print(f"❌ Backtest log not found: {log_path}")
        sys.exit(1)

    with open(log_path, "r") as f:
        log_content = f.read()
        if "funnel" not in log_content.lower():
            print("❌ Funnel line not present in backtest.log")
            sys.exit(1)
        print("✅ Funnel line present in backtest.log")

    print("✅ Sprint 6 done")

if __name__ == "__main__":
    main()