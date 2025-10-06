"""Deterministic time-of-day and calendar features."""

from __future__ import annotations

import numpy as np
import pandas as pd

_RTH_MINUTES = 390

_SESSION_BUCKETS = {
    "session_open": (0, 60),  # first hour
    "session_lunch": (150, 240),
    "session_power": (300, _RTH_MINUTES),
}


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Append deterministic seasonal features without mutating *frame*."""
    if frame.empty:
        return frame.copy()

    data = frame.copy()
    timestamps = pd.to_datetime(data["timestamp"], utc=True).dt.tz_convert("America/New_York")

    minute_of_day = _minute_of_day(timestamps)
    minute_progress = minute_of_day.astype(float) / float(_RTH_MINUTES - 1)

    data["minute_of_day"] = minute_of_day
    data["minute_index"] = minute_of_day
    data["minute_progress"] = minute_progress

    for harmonic in (1, 2, 3):
        angle = 2.0 * np.pi * float(harmonic) * minute_progress
        data[f"fourier_sin_{harmonic}"] = np.sin(angle)
        data[f"fourier_cos_{harmonic}"] = np.cos(angle)

    for bucket, (start, end) in _SESSION_BUCKETS.items():
        data[bucket] = ((minute_of_day >= start) & (minute_of_day < end)).astype(int)

    data["day_of_week"] = timestamps.dt.dayofweek
    data["is_month_end"] = timestamps.dt.is_month_end.astype(int)
    return data


def _minute_of_day(timestamps: pd.Series) -> np.ndarray:
    minutes = timestamps.dt.hour.astype(int) * 60 + timestamps.dt.minute.astype(int) - (9 * 60 + 30)
    minutes = minutes.clip(lower=0, upper=_RTH_MINUTES - 1)
    return minutes.to_numpy()


__all__ = ["add_time_features"]
