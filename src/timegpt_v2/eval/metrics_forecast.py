"""Forecast evaluation metrics (stub)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ForecastMetrics:
    rmae: float
    rrmse: float
