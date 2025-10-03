"""Synthetic data generators used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


@dataclass
class SyntheticConfig:
    symbol: str = "TEST"
    start: datetime = datetime(2024, 1, 1, 9, 30)
    periods: int = 10
    freq: str = "1min"


def generate_bars(config: SyntheticConfig | None = None) -> pd.DataFrame:
    cfg = config or SyntheticConfig()
    index = pd.date_range(start=cfg.start, periods=cfg.periods, freq=cfg.freq, tz="America/New_York")
    data = {
        "open": 100.0,
        "high": 100.5,
        "low": 99.5,
        "close": 100.2,
        "volume": 1000,
    }
    frame = pd.DataFrame(data, index=index)
    frame.index.name = "timestamp"
    frame["symbol"] = cfg.symbol
    return frame.reset_index()
