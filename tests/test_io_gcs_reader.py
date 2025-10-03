from __future__ import annotations

from datetime import datetime

import pandas as pd

from timegpt_v2.io.gcs_reader import GCSReader, ReaderConfig


def _write_parquet(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_read_month_normalises_aliases(tmp_path):
    timestamps = pd.date_range("2024-07-01 09:25", periods=6, freq="1min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "ts": timestamps,
            "o": [100, 101, 102, 103, 104, 105],
            "h": [101, 102, 103, 104, 105, 106],
            "l": [99, 100, 101, 102, 103, 104],
            "c": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "v": [1000, 1100, 1200, 1300, 1400, 1500],
            "ticker": ["AAPL"] * 6,
            "adj_close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
            "adj_open": [100, 101, 102, 103, 104, 105],
            "adj_high": [101, 102, 103, 104, 105, 106],
            "adj_low": [99, 100, 101, 102, 103, 104],
        }
    )
    target = tmp_path / "stocks" / "AAPL" / "2024" / "AAPL_2024_07.parquet"
    _write_parquet(target, frame)

    reader = GCSReader(
        ReaderConfig(
            bucket=str(tmp_path), template="stocks/{ticker}/{yyyy}/{ticker}_{yyyy_mm}.parquet"
        )
    )
    result = reader.read_month("AAPL", 2024, 7)

    assert set(["timestamp", "open", "high", "low", "close", "volume", "symbol"]).issubset(
        result.columns
    )
    assert result["timestamp"].dt.tz is not None
    assert result["timestamp"].dt.time.min().strftime("%H:%M") == "09:30"
    assert result["timestamp"].dt.time.max().strftime("%H:%M") == "09:30"
    assert (result["symbol"].unique() == ["AAPL"]).all()


def test_read_universe_filters_to_requested_range(tmp_path):
    july_idx = pd.date_range("2024-07-01 09:30", periods=2, freq="1min", tz="America/New_York")
    aug_idx = pd.date_range("2024-08-01 09:30", periods=2, freq="1min", tz="America/New_York")
    july = pd.DataFrame(
        {
            "timestamp": july_idx,
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
            "symbol": ["AAPL", "AAPL"],
            "adj_open": [100, 101],
            "adj_high": [101, 102],
            "adj_low": [99, 100],
            "adj_close": [100.5, 101.5],
        }
    )
    august = july.copy()
    august["timestamp"] = aug_idx

    _write_parquet(tmp_path / "stocks" / "AAPL" / "2024" / "AAPL_2024_07.parquet", july)
    _write_parquet(tmp_path / "stocks" / "AAPL" / "2024" / "AAPL_2024_08.parquet", august)

    reader = GCSReader(
        ReaderConfig(
            bucket=str(tmp_path), template="stocks/{ticker}/{yyyy}/{ticker}_{yyyy_mm}.parquet"
        )
    )
    subset = reader.read_universe(
        ["AAPL"], datetime(2024, 7, 1).date(), datetime(2024, 7, 31).date()
    )

    assert subset["timestamp"].dt.month.unique().tolist() == [7]
    assert len(subset) == 2
