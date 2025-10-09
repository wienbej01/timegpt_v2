from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

ET_ZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class PhaseConfig:
    """Logical buckets for backtest evaluation windows."""

    in_sample: tuple[str, ...]
    oos: tuple[str, ...]
    stress: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Sequence[str]] | None) -> PhaseConfig:
        payload = payload or {}
        return cls(
            in_sample=tuple(str(month) for month in payload.get("in_sample", ())),
            oos=tuple(str(month) for month in payload.get("oos", ())),
            stress=tuple(str(month) for month in payload.get("stress", ())),
        )

    def lookup(self, month: str) -> str:
        if month in self.oos:
            return "oos"
        if month in self.stress:
            return "stress"
        return "in_sample"


def assign_phases(
    trades: pd.DataFrame,
    *,
    config: PhaseConfig,
) -> pd.DataFrame:
    """Annotate each trade with its regime (IS, OOS, stress)."""
    if trades.empty:
        enriched = trades.copy()
        enriched["phase"] = pd.Series(dtype="string")
        return enriched

    working = trades.copy()
    working["entry_ts"] = pd.to_datetime(working["entry_ts"], utc=True)
    working["trade_month"] = working["entry_ts"].dt.to_period("M").astype(str)
    working["phase"] = working["trade_month"].map(config.lookup)
    return working


def compute_portfolio_summaries(
    trades: pd.DataFrame,
    *,
    aggregation: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return portfolio-level and per-symbol summaries."""
    if aggregation != "equal_weight":
        raise ValueError(f"Unsupported portfolio aggregation: {aggregation}")

    if trades.empty:
        empty_cols = [
            "phase",
            "level",
            "trade_count",
            "total_net_pnl",
            "total_gross_pnl",
            "hit_rate",
            "sharpe",
        ]
        return (
            pd.DataFrame(columns=empty_cols),
            pd.DataFrame(columns=[*empty_cols[:-2], "symbol"]),
        )

    working = trades.copy()
    working["entry_ts"] = pd.to_datetime(working["entry_ts"], utc=True)
    working["exit_ts"] = pd.to_datetime(working["exit_ts"], utc=True)

    portfolio_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []

    _append_symbol_rows(working, symbol_rows)
    _append_portfolio_rows(working, portfolio_rows)

    portfolio_df = pd.DataFrame(portfolio_rows)
    symbol_df = pd.DataFrame(symbol_rows)
    portfolio_df.sort_values(["phase", "level"], inplace=True)
    symbol_df.sort_values(["phase", "symbol"], inplace=True)
    return portfolio_df.reset_index(drop=True), symbol_df.reset_index(drop=True)


def _append_symbol_rows(trades: pd.DataFrame, destination: list[dict[str, Any]]) -> None:
    grouped = trades.groupby(["phase", "symbol"], sort=True)
    for (phase, symbol), group in grouped:
        trade_count = int(len(group))
        net_pnl = float(group["net_pnl"].sum())
        gross_pnl = float(group["gross_pnl"].sum())
        hit_rate = float((group["net_pnl"] > 0).mean()) if trade_count else 0.0
        destination.append(
            {
                "phase": phase,
                "level": "symbol",
                "symbol": symbol,
                "trade_count": trade_count,
                "total_net_pnl": net_pnl,
                "total_gross_pnl": gross_pnl,
                "hit_rate": hit_rate,
            }
        )


def _append_portfolio_rows(trades: pd.DataFrame, destination: list[dict[str, Any]]) -> None:
    trades = trades.copy()
    trades["entry_date"] = trades["entry_ts"].dt.tz_convert(ET_ZONE).dt.date

    for phase, phase_group in trades.groupby("phase", sort=True):
        trade_count = int(len(phase_group))
        total_net = float(phase_group["net_pnl"].sum())
        total_gross = float(phase_group["gross_pnl"].sum())
        hit_rate = float((phase_group["net_pnl"] > 0).mean()) if trade_count else 0.0

        daily = (
            phase_group.groupby(["entry_date", "symbol"], sort=True)["net_pnl"].sum().reset_index()
        )

        if daily.empty:
            sharpe = 0.0
        else:
            pivot = daily.pivot_table(
                index="entry_date",
                columns="symbol",
                values="net_pnl",
                fill_value=0.0,
            )
            portfolio_returns = pivot.mean(axis=1)
            sharpe = _sharpe_ratio(portfolio_returns)

        destination.append(
            {
                "phase": phase,
                "level": "portfolio",
                "trade_count": trade_count,
                "total_net_pnl": total_net,
                "total_gross_pnl": total_gross,
                "hit_rate": hit_rate,
                "sharpe": sharpe,
            }
        )


def _sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    stdev = returns.std(ddof=0)
    if math.isclose(stdev, 0.0):
        return 0.0
    annualizer = math.sqrt(252.0)
    return float(returns.mean() / stdev * annualizer)


def compute_cost_scenarios(
    trades: pd.DataFrame,
    *,
    multipliers: Sequence[float],
    aggregation: str,
) -> pd.DataFrame:
    """Evaluate portfolio behaviour under alternative cost multipliers."""
    if aggregation != "equal_weight":
        raise ValueError(f"Unsupported portfolio aggregation: {aggregation}")

    if trades.empty:
        return pd.DataFrame(
            columns=[
                "cost_multiplier",
                "total_net_pnl",
                "trade_count",
                "hit_rate",
                "sharpe",
            ]
        )

    working = trades.copy()
    working["entry_ts"] = pd.to_datetime(working["entry_ts"], utc=True)
    working["entry_date"] = working["entry_ts"].dt.tz_convert(ET_ZONE).dt.date

    gross = working["gross_pnl"]
    base_cost = (
        (working["costs_bp"].astype(float) / 10_000.0)
        * working["entry_price"].astype(float).abs()
        * working["position"].abs()
        * 2.0
    )

    rows: list[dict[str, Any]] = []
    for multiplier in multipliers:
        net = gross - base_cost * float(multiplier)

        trade_count = int(len(net))
        total_net = float(net.sum())
        hit_rate = float((net > 0).mean()) if trade_count else 0.0

        daily = (
            working.assign(net_multi=net)
            .groupby(["entry_date", "symbol"], sort=True)["net_multi"]
            .sum()
            .reset_index()
        )

        if daily.empty:
            sharpe = 0.0
        else:
            pivot = daily.pivot_table(
                index="entry_date",
                columns="symbol",
                values="net_multi",
                fill_value=0.0,
            )
            returns = pivot.mean(axis=1)
            sharpe = _sharpe_ratio(returns)

        rows.append(
            {
                "cost_multiplier": float(multiplier),
                "total_net_pnl": total_net,
                "trade_count": trade_count,
                "hit_rate": hit_rate,
                "sharpe": sharpe,
            }
        )

    return pd.DataFrame(rows).sort_values("cost_multiplier").reset_index(drop=True)
