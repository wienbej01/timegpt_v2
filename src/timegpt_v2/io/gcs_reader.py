"""GCS reader utilities for intraday bar data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

try:  # pragma: no cover - optional import for runtime
    import gcsfs  # type: ignore
except ImportError:  # pragma: no cover - handled in __post_init__
    gcsfs = None  # type: ignore

import fsspec

ET_ZONE = ZoneInfo("America/New_York")

_COLUMN_ALIASES = {
    "timestamp": {"t", "ts", "timestamp", "time"},
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "c"},
    "volume": {"volume", "v"},
    "symbol": {"symbol", "ticker"},
    "adj_open": {"adj_open", "open_adj"},
    "adj_high": {"adj_high", "high_adj"},
    "adj_low": {"adj_low", "low_adj"},
    "adj_close": {"adj_close", "close_adj", "adjusted_close"},
}

_RTH_START = time(9, 30)
_RTH_END = time(16, 0)


@dataclass
class ReaderConfig:
    bucket: str
    template: str


class GCSReader:
    """Read intraday bars stored as parquet files on GCS or local filesystem."""

    def __init__(self, config: ReaderConfig, fs: fsspec.AbstractFileSystem | None = None) -> None:
        self._config = config
        self._fs = fs or self._build_filesystem(config.bucket)
        self._base_path = self._derive_base_path(config.bucket)

    def read_month(self, ticker: str, year: int, month: int) -> pd.DataFrame:
        """Read a single ticker/month parquet and apply canonical normalisation."""
        relative_path = self._config.template.format(
            ticker=ticker,
            yyyy=str(year),
            yyyy_mm=f"{year}_{month:02d}",
        )
        full_path = self._join_path(relative_path)
        with self._fs.open(full_path) as fh:
            frame = pd.read_parquet(fh)
        frame = self._normalise_dataframe(frame, ticker)
        return frame

    def read_range(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Read and concatenate all months covering the inclusive date range."""
        months = list(_iter_months(start, end))
        if not months:
            return pd.DataFrame()
        frames = [self.read_month(ticker, year, month) for year, month in months]
        combined = pd.concat(frames, ignore_index=True)
        mask = (combined["timestamp"].dt.date >= start) & (combined["timestamp"].dt.date <= end)
        return combined.loc[mask].reset_index(drop=True)

    def read_universe(self, tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
        """Read multiple tickers across date range into a single DataFrame."""
        frames = [self.read_range(ticker, start, end) for ticker in tickers]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["symbol", "timestamp"])
            .reset_index(drop=True)
        )

    def _join_path(self, relative: str) -> str:
        if self._base_path is None:
            return relative
        if isinstance(self._base_path, Path):
            return str(self._base_path / relative)
        return f"{self._base_path}/{relative}".replace("//", "/")

    @staticmethod
    def _build_filesystem(bucket: str) -> fsspec.AbstractFileSystem:
        path = Path(bucket)
        if path.exists():
            return fsspec.filesystem("file")
        if gcsfs is None:
            raise RuntimeError("gcsfs is required to access remote GCS buckets")
        return gcsfs.GCSFileSystem()

    @staticmethod
    def _derive_base_path(bucket: str) -> Path | str | None:
        path = Path(bucket)
        if path.exists():
            return path
        return bucket.strip("/") or None

    @staticmethod
    def _normalise_dataframe(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        renamed = _rename_columns(frame)
        ticker_str = str(ticker)
        if "symbol" in renamed.columns:
            renamed["symbol"] = renamed["symbol"].fillna(ticker_str).astype(str)
        else:
            renamed["symbol"] = ticker_str
        renamed["timestamp"] = pd.to_datetime(renamed["timestamp"], utc=False, errors="raise")
        if renamed["timestamp"].dt.tz is None:
            renamed["timestamp"] = renamed["timestamp"].dt.tz_localize(ET_ZONE)
        else:
            renamed["timestamp"] = renamed["timestamp"].dt.tz_convert(ET_ZONE)
        renamed = _filter_rth(renamed)
        renamed = renamed.sort_values("timestamp").reset_index(drop=True)
        return renamed


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in frame.columns:
                rename_map[alias] = canonical
    renamed = frame.rename(columns=rename_map).copy()
    return renamed


def _filter_rth(frame: pd.DataFrame) -> pd.DataFrame:
    timestamps = frame["timestamp"].dt.tz_convert(ET_ZONE)
    mask = (timestamps.dt.time >= _RTH_START) & (timestamps.dt.time < _RTH_END)
    return frame.loc[mask].copy()


def _iter_months(start: date, end: date) -> Iterable[tuple[int, int]]:
    if start > end:
        return []
    cursor = date(start.year, start.month, 1)
    end_marker = date(end.year, end.month, 1)
    months: list[tuple[int, int]] = []
    while cursor <= end_marker:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    return months
