"""Client for interacting with TimeGPT (stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ForecastResult:
    data: pd.DataFrame


class TimeGPTClient:
    def forecast(self, payload: dict[str, Any]) -> ForecastResult:
        raise NotImplementedError
