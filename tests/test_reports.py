from __future__ import annotations

from pathlib import Path

import pandas as pd

from timegpt_v2.reports.builder import build_report


def test_robustness_report_contains_sections(tmp_path: Path) -> None:
    run_dir = tmp_path
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    portfolio_summary = pd.DataFrame(
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
                "total_net_pnl": 0.7,
                "trade_count": 5,
                "hit_rate": 0.5,
                "sharpe": 0.5,
            },
        ]
    )

    portfolio_summary.to_csv(eval_dir / "portfolio_summary.csv", index=False)
    cost_table.to_csv(eval_dir / "cost_sensitivity.csv", index=False)

    report_path = build_report(run_dir)

    contents = report_path.read_text(encoding="utf-8")
    assert "## Out-of-Sample Portfolio" in contents
    assert "## Cost Sensitivity" in contents
    assert "| cost_multiplier |" in contents
