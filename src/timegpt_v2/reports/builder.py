"""Sprint 9 reporting utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def build_report(run_dir: Path, *, config_dir: Path | None = None) -> Path:
    """Render the Sprint 9 operational report for a completed run."""

    eval_dir = run_dir / "eval"
    portfolio_path = eval_dir / "portfolio_summary.csv"
    cost_path = eval_dir / "cost_sensitivity.csv"
    forecast_path = eval_dir / "forecast_metrics.csv"
    reliability_path = eval_dir / "pit_reliability.csv"
    grid_summary_path = eval_dir / "grid" / "summary.csv"

    required = [portfolio_path, cost_path, forecast_path, reliability_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation artifacts: " + ", ".join(missing))

    portfolio = pd.read_csv(portfolio_path)
    cost_table = pd.read_csv(cost_path)
    forecast_metrics = pd.read_csv(forecast_path)
    reliability = pd.read_csv(reliability_path)

    grid_table = None
    if grid_summary_path.exists():
        grid_table = pd.read_csv(grid_summary_path)

    meta = _load_meta(run_dir)
    config_dir = config_dir or Path(meta.get("config_dir", "configs"))

    universe_cfg = _safe_load_yaml(config_dir / "universe.yaml")
    forecast_cfg = _safe_load_yaml(config_dir / "forecast.yaml")
    trading_cfg = _safe_load_yaml(config_dir / "trading.yaml")
    backtest_cfg = _safe_load_yaml(config_dir / "backtest.yaml")

    sections = [
        _build_header_section(run_dir, universe_cfg),
        _build_configuration_section(forecast_cfg, trading_cfg, backtest_cfg),
        _build_forecast_section(forecast_metrics, reliability),
        _build_portfolio_section(portfolio),
        _build_cost_section(cost_table),
        _build_grid_section(grid_table),
        _build_artifact_section(
            meta,
            {
                "portfolio": portfolio_path,
                "cost": cost_path,
                "forecast": forecast_path,
                "reliability": reliability_path,
                "grid_summary": grid_summary_path if grid_summary_path.exists() else None,
            },
        ),
    ]

    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
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


def _build_header_section(run_dir: Path, universe_cfg: dict[str, Any]) -> str:
    run_id = run_dir.name
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    tickers = universe_cfg.get("tickers", [])
    date_cfg = universe_cfg.get("dates", {})
    start_date = date_cfg.get("start", "N/A")
    end_date = date_cfg.get("end", "N/A")
    tz = universe_cfg.get("tz", "America/New_York")

    return "\n".join(
        [
            "# TimeGPT Intraday v2 Report",
            "",
            f"- **Run ID:** `{run_id}`",
            f"- **Generated:** `{generated_at}`",
            f"- **Universe:** `{', '.join(str(t) for t in tickers)}`",
            f"- **Date Range:** `{start_date}` → `{end_date}` ({tz})",
            "",
        ]
    )


def _build_configuration_section(
    forecast_cfg: dict[str, Any],
    trading_cfg: dict[str, Any],
    backtest_cfg: dict[str, Any],
) -> str:
    horizon = forecast_cfg.get("horizon_min", "?")
    snapshots = ", ".join(str(s) for s in forecast_cfg.get("snapshots_et", []))
    k_sigma = trading_cfg.get("k_sigma")
    s_stop = trading_cfg.get("s_stop")
    s_take = trading_cfg.get("s_take")
    cost_multipliers = backtest_cfg.get("cost_multipliers", [])
    aggregation = backtest_cfg.get("portfolio_aggregation", "?")

    return "\n".join(
        [
            "## Configuration Highlights",
            "",
            f"- Forecast horizon: `{horizon}` minutes with snapshots at `{snapshots}`",
            f"- Trading thresholds: `k_sigma={k_sigma}`, `s_stop={s_stop}`, `s_take={s_take}`",
            f"- Portfolio aggregation: `{aggregation}`; cost multipliers: `{cost_multipliers}`",
            "",
        ]
    )


def _build_forecast_section(
    forecast_metrics: pd.DataFrame,
    reliability: pd.DataFrame,
) -> str:
    median_rmae = float(forecast_metrics["rmae"].median())
    median_rrmse = float(forecast_metrics["rrmse"].median())
    median_pit = float(forecast_metrics["pit_coverage"].median())
    forecast_pass = median_rmae < 0.95 and median_rrmse < 0.97
    calibration_pass = abs(median_pit - 0.5) <= 0.02

    summary_table = forecast_metrics[
        [
            "symbol",
            "count",
            "mae",
            "rmse",
            "rmae",
            "rrmse",
            "pit_coverage",
        ]
    ].copy()

    gauge_lines = [
        "- Forecast gates: "
        f"{'PASS' if forecast_pass else 'FAIL'} "
        f"(median rMAE={median_rmae:.3f}, median rRMSE={median_rrmse:.3f})",
        "- Calibration: "
        f"{'PASS' if calibration_pass else 'FAIL'} "
        f"(median PIT={median_pit:.3f})",
    ]

    reliability_present = not reliability.empty

    return "\n".join(
        [
            "## Forecast KPIs",
            "",
            *gauge_lines,
            "",
            _markdown_table(summary_table),
            "",
            (
                "Calibration bins available in `eval/pit_reliability.csv`."
                if reliability_present
                else "Calibration data unavailable."
            ),
            "",
        ]
    )


def _build_grid_section(grid_table: pd.DataFrame | None) -> str:
    if grid_table is None or grid_table.empty:
        return "\n".join(["## Grid Search", "", "No sweep artifacts were found.", ""])

    top_row = grid_table.sort_values("rank").iloc[0].to_dict()
    raw_grid_point = top_row.get("grid_point", "{}")
    if isinstance(raw_grid_point, str):
        grid_point = json.loads(raw_grid_point)
    else:
        grid_point = raw_grid_point or {}
    details = {
        "hash": top_row.get("combo_hash"),
        "total_net_pnl": _format_number(top_row.get("total_net_pnl", 0.0)),
        "hit_rate": _format_number(top_row.get("hit_rate", 0.0)),
    }

    return "\n".join(
        [
            "## Grid Search",
            "",
            "Top-ranked parameter set:",
            "",
            f"- Parameters: `{grid_point}`",
            f"- Combo hash: `{details['hash']}`",
            f"- Net P&L: `{details['total_net_pnl']}`",
            f"- Hit rate: `{details['hit_rate']}`",
            "",
        ]
    )


def _build_artifact_section(meta: dict[str, Any], paths: dict[str, Path | None]) -> str:
    lines = ["## Key Artifacts", ""]
    for label, value in paths.items():
        if value is None:
            continue
        lines.append(f"- {label.replace('_', ' ').title()}: `{value}`")

    meta_path = meta.get("meta_path")
    if meta_path:
        lines.append(f"- Meta: `{meta_path}`")
    lines.append("")
    return "\n".join(lines)


def _load_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        return {"meta_path": str(meta_path)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["meta_path"] = str(meta_path)
    return meta


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


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
