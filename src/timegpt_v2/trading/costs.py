"""Trading cost models (stub)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    fee_bps: float
