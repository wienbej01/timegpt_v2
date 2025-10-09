"""Robustness reporting utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_report(run_dir: Path) -> Path:
    """Render the Sprint 8 robustness report for a completed run."""
    eval_dir = run_dir / "eval"
    portfolio_path = eval_dir / "portfolio_summary.csv"
    cost_path = eval_dir / "cost_sensitivity.csv"

    if not portfolio_path.exists() or not cost_path.exists():
        missing = []
        if not portfolio_path.exists():
            missing.append(str(portfolio_path))
        if not cost_path.exists():
            missing.append(str(cost_path))
        raise FileNotFoundError("Missing evaluation artifacts: " + ", ".join(missing))

    portfolio = pd.read_csv(portfolio_path)
    cost_table = pd.read_csv(cost_path)

    portfolio_section = _build_portfolio_section(portfolio)
    cost_section = _build_cost_section(cost_table)

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "robustness_report.md"
    report_path.write_text("\n".join([portfolio_section, cost_section]) + "\n", encoding="utf-8")
    return report_path


def _build_portfolio_section(portfolio: pd.DataFrame) -> str:
    relevant = portfolio[portfolio["level"] == "portfolio"].copy()
    if relevant.empty:
        relevant = pd.DataFrame(
            [
                {
                    "phase": "oos",
                    "trade_count": 0,
                    "total_net_pnl": 0.0,
                    "hit_rate": 0.0,
                    "sharpe": 0.0,
                }
            ]
        )
    columns = ["phase", "trade_count", "total_net_pnl", "hit_rate", "sharpe"]
    display = relevant.loc[:, columns]
    return "\n".join(
        [
            "# Robustness Report",
            "",
            "## Out-of-Sample Portfolio",
            "",
            _markdown_table(display),
            "",
        ]
    )


def _build_cost_section(cost_table: pd.DataFrame) -> str:
    if cost_table.empty:
        cost_table = pd.DataFrame(
            [
                {
                    "cost_multiplier": 1.0,
                    "total_net_pnl": 0.0,
                    "trade_count": 0,
                    "hit_rate": 0.0,
                    "sharpe": 0.0,
                }
            ]
        )
    return "\n".join(
        [
            "## Cost Sensitivity",
            "",
            _markdown_table(cost_table),
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no data)"

    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(_format_number)
        else:
            formatted[column] = formatted[column].astype(str)

    header = "| " + " | ".join(formatted.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(formatted.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in formatted.to_numpy(dtype=str)]
    return "\n".join([header, divider, *rows])


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{float(value):.4f}"
