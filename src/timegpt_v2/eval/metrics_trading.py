from __future__ import annotations

import numpy as np
import pandas as pd


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
