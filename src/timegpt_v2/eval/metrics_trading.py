"""Trading evaluation metrics (stub)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradingMetrics:
    sharpe: float
    hit_rate: float
