from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from timegpt_v2.reports.builder import build_report


def _write_yaml(path: Path, payload: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_build_report_generates_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    grid_dir = eval_dir / "grid"
    eval_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "trades").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)

    portfolio = pd.DataFrame(
        [
            {
                "phase": "in_sample",
                "level": "portfolio",
                "trade_count": 5,
                "total_net_pnl": 1.2,
                "total_gross_pnl": 1.5,
                "hit_rate": 0.6,
                "sharpe": 0.8,
            },
            {
                "phase": "oos",
                "level": "portfolio",
                "trade_count": 3,
                "total_net_pnl": 0.9,
                "total_gross_pnl": 1.1,
                "hit_rate": 0.67,
                "sharpe": 0.75,
            },
        ]
    )
    portfolio.to_csv(eval_dir / "portfolio_summary.csv", index=False)

    cost_table = pd.DataFrame(
        [
            {
                "cost_multiplier": 1.0,
                "total_net_pnl": 1.2,
                "trade_count": 5,
                "hit_rate": 0.6,
                "sharpe": 0.8,
            },
            {
                "cost_multiplier": 1.5,
                "total_net_pnl": 0.9,
                "trade_count": 5,
                "hit_rate": 0.58,
                "sharpe": 0.6,
            },
        ]
    )
    cost_table.to_csv(eval_dir / "cost_sensitivity.csv", index=False)

    forecast_metrics = pd.DataFrame(
        [
            {
                "symbol": "SYN",
                "count": 10,
                "mae": 0.12,
                "rmse": 0.2,
                "rmae": 0.8,
                "rrmse": 0.85,
                "pinball_loss_q25": 0.07,
                "pinball_loss_q75": 0.06,
                "pit_coverage": 0.51,
            }
        ]
    )
    forecast_metrics.to_csv(eval_dir / "forecast_metrics.csv", index=False)

    reliability = pd.DataFrame(
        {
            "symbol": ["SYN"] * 3,
            "bin": [0, 1, 2],
            "prob_pred": [0.1, 0.5, 0.9],
            "prob_true": [0.12, 0.52, 0.88],
        }
    )
    reliability.to_csv(eval_dir / "pit_reliability.csv", index=False)

    grid_summary = pd.DataFrame(
        [
            {
                "grid_point": json.dumps({"k_sigma": 0.5, "s_stop": 1.0, "s_take": 1.0}),
                "combo_hash": "abc123",
                "total_net_pnl": 1.0,
                "hit_rate": 0.6,
                "rank": 1,
            }
        ]
    )
    grid_summary.to_csv(grid_dir / "summary.csv", index=False)

    meta = {
        "config_dir": str(tmp_path / "configs"),
        "steps": {
            "evaluate": {
                "forecast_metrics_path": str(eval_dir / "forecast_metrics.csv"),
                "trading_metrics_path": str(eval_dir / "bt_summary.csv"),
                "cost_sensitivity_path": str(eval_dir / "cost_sensitivity.csv"),
                "reliability_path": str(eval_dir / "pit_reliability.csv"),
                "median_rmae": 0.8,
                "median_rrmse": 0.85,
                "median_pit_coverage": 0.51,
            }
        },
        "command": "report",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    configs_dir = tmp_path / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        configs_dir / "universe.yaml",
        {
            "tickers": ["SYN"],
            "dates": {"start": "2024-07-01", "end": "2024-08-31"},
            "tz": "America/New_York",
        },
    )
    _write_yaml(
        configs_dir / "forecast.yaml",
        {"horizon_min": 15, "snapshots_et": ["10:00", "14:30"]},
    )
    _write_yaml(
        configs_dir / "trading.yaml",
        {"k_sigma": [0.5], "s_stop": [1.0], "s_take": [1.0]},
    )
    _write_yaml(
        configs_dir / "backtest.yaml",
        {"cost_multipliers": [1.0, 1.5], "portfolio_aggregation": "equal_weight"},
    )

    report_path = build_report(run_dir, config_dir=configs_dir)

    contents = report_path.read_text(encoding="utf-8")
    assert "# TimeGPT Intraday v2 Report" in contents
    assert "## Forecast KPIs" in contents
    assert "## Grid Search" in contents
    assert "abc123" in contents
