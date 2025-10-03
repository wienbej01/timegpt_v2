"""Synthetic data generators used in tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET_ZONE = ZoneInfo("America/New_York")


@dataclass
class SyntheticConfig:
    symbol: str = "TEST"
    start: datetime = datetime(2024, 1, 2, 9, 30, tzinfo=ET_ZONE)
    sessions: int = 1
    minutes_per_session: int = 390
    seed: int = 7


def _session_index(start: datetime, minutes: int) -> pd.DatetimeIndex:
    if start.time() != time(9, 30):
        raise ValueError("Synthetic sessions must start at 09:30 ET")
    return pd.date_range(start=start, periods=minutes, freq="1min", tz=ET_ZONE)


def generate_bars(config: SyntheticConfig | None = None) -> pd.DataFrame:
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    all_frames: list[pd.DataFrame] = []
    session_start = cfg.start
    for _ in range(cfg.sessions):
        index = _session_index(session_start, cfg.minutes_per_session)
        base_price = 100 + rng.normal(0, 0.5)
        close = base_price + rng.normal(0, 0.3, size=len(index)).cumsum()
        open_ = close + rng.normal(0, 0.05, size=len(index))
        high = np.maximum(open_, close) + rng.uniform(0, 0.1, size=len(index))
        low = np.minimum(open_, close) - rng.uniform(0, 0.1, size=len(index))
        volume = rng.integers(900, 1100, size=len(index))
        frame = pd.DataFrame(
            {
                "timestamp": index,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "adj_open": open_,
                "adj_high": high,
                "adj_low": low,
                "adj_close": close,
            }
        )
        frame["symbol"] = cfg.symbol
        all_frames.append(frame)
        session_start = (session_start + pd.Timedelta(days=1)).replace(hour=9, minute=30)
    return pd.concat(all_frames, ignore_index=True)
