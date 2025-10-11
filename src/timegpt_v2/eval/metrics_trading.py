from __future__ import annotations

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET_ZONE = ZoneInfo("America/New_York")


def sharpe_ratio(pnl: pd.Series, trading_days: int) -> float:
    """Calculate the Sharpe ratio."""
    if not isinstance(pnl.index, pd.DatetimeIndex):
        raise TypeError("pnl.index must be a DatetimeIndex")
    daily_pnl = pnl.groupby(pnl.index.date).sum()
    if daily_pnl.std() == 0:
        return 0.0
    return daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)


def max_drawdown(pnl: pd.Series) -> float:
    """Calculate the maximum drawdown."""
    cumulative_pnl = pnl.cumsum()
    peak = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - peak) / peak
    return drawdown.min()


def hit_rate(pnl: pd.Series) -> float:
    """Calculate the hit rate."""
    return (pnl > 0).mean()


def portfolio_sharpe(trades: pd.DataFrame, *, phase: str | None = None) -> float:
    """Calculate portfolio Sharpe ratio from trades DataFrame."""
    working = trades.copy()
    if phase:
        working = working[working["phase"] == phase]
    if working.empty:
        return 0.0

    working["entry_ts"] = pd.to_datetime(working["entry_ts"], utc=True)
    working["entry_date"] = working["entry_ts"].dt.tz_convert(ET_ZONE).dt.date

    daily = (
        working.groupby(["entry_date", "symbol"], sort=True)["net_pnl"].sum().reset_index()
    )

    if daily.empty:
        return 0.0

    pivot = daily.pivot_table(
        index="entry_date",
        columns="symbol",
        values="net_pnl",
        fill_value=0.0,
    )
    portfolio_returns = pivot.mean(axis=1)
    stdev = portfolio_returns.std(ddof=0)
    if math.isclose(stdev, 0.0):
        return 0.0
    annualizer = math.sqrt(252.0)
    return float(portfolio_returns.mean() / stdev * annualizer)


def portfolio_max_drawdown(trades: pd.DataFrame, *, phase: str | None = None) -> float:
    """Calculate portfolio maximum drawdown from trades DataFrame."""
    working = trades.copy()
    if phase:
        working = working[working["phase"] == phase]
    if working.empty:
        return 0.0

    working["entry_ts"] = pd.to_datetime(working["entry_ts"], utc=True)
    working["entry_date"] = working["entry_ts"].dt.tz_convert(ET_ZONE).dt.date

    daily = (
        working.groupby(["entry_date", "symbol"], sort=True)["net_pnl"].sum().reset_index()
    )

    if daily.empty:
        return 0.0

    pivot = daily.pivot_table(
        index="entry_date",
        columns="symbol",
        values="net_pnl",
        fill_value=0.0,
    )
    portfolio_returns = pivot.mean(axis=1)
    cumulative = portfolio_returns.cumsum()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def portfolio_hit_rate(trades: pd.DataFrame, *, phase: str | None = None) -> float:
    """Calculate portfolio hit rate from trades DataFrame."""
    working = trades.copy()
    if phase:
        working = working[working["phase"] == phase]
    if working.empty:
        return 0.0
    return float((working["net_pnl"] > 0).mean())


def portfolio_total_pnl(trades: pd.DataFrame, *, phase: str | None = None) -> float:
    """Calculate portfolio total net PnL from trades DataFrame."""
    working = trades.copy()
    if phase:
        working = working[working["phase"] == phase]
    if working.empty:
        return 0.0
    return float(working["net_pnl"].sum())


def per_symbol_metrics(trades: pd.DataFrame, *, phase: str | None = None) -> pd.DataFrame:
    """Compute per-symbol KPIs from trades DataFrame."""
    working = trades.copy()
    if phase:
        working = working[working["phase"] == phase]

    if working.empty:
        return pd.DataFrame(columns=["symbol", "trade_count", "total_net_pnl", "hit_rate", "sharpe", "max_drawdown"])

    working["entry_ts"] = pd.to_datetime(working["entry_ts"], utc=True)
    working["entry_date"] = working["entry_ts"].dt.tz_convert(ET_ZONE).dt.date

    rows = []
    for symbol, group in working.groupby("symbol"):
        trade_count = len(group)
        total_net_pnl = float(group["net_pnl"].sum())
        hit_rate = float((group["net_pnl"] > 0).mean()) if trade_count else 0.0

        daily = group.groupby("entry_date")["net_pnl"].sum()
        if daily.empty or daily.std() == 0:
            sharpe = 0.0
        else:
            sharpe = float(daily.mean() / daily.std() * np.sqrt(252))

        cumulative = daily.cumsum()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_dd = float(drawdown.min())

        rows.append({
            "symbol": symbol,
            "trade_count": trade_count,
            "total_net_pnl": total_net_pnl,
            "hit_rate": hit_rate,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
        })

    return pd.DataFrame(rows)
