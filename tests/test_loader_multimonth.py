from __future__ import annotations

import logging
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from timegpt_v2.loader.gcs_loader import enumerate_month_uris, load_history


def test_enumerate_month_uris():
    uris = list(
        enumerate_month_uris(
            template="stocks/{ticker}/{yyyy}/{ticker}_{yyyy_mm}.parquet",
            start_dt=date(2024, 6, 5),
            end_dt=date(2024, 7, 5),
            symbol="AAPL",
        )
    )
    assert uris == [
        "stocks/AAPL/2024/AAPL_2024_06.parquet",
        "stocks/AAPL/2024/AAPL_2024_07.parquet",
    ]


@patch("pandas.read_parquet")
def test_load_history(mock_read_parquet, caplog):
    mock_read_parquet.side_effect = [
        pd.DataFrame({"timestamp": pd.to_datetime(["2024-06-25"], utc=True), "close": [100]}),
        pd.DataFrame({"timestamp": pd.to_datetime(["2024-07-15"], utc=True), "close": [110]}),
    ]

    gcs_config = {
        "bucket": "test-bucket",
        "template": "stocks/{ticker}/{yyyy}/{ticker}_{yyyy_mm}.parquet",
    }
    logger = logging.getLogger(__name__)

    with caplog.at_level(logging.INFO):
        df = load_history(
            symbol="AAPL",
            start=date(2024, 7, 20),
            end=date(2024, 7, 20),
            rolling_history_days=30,
            gcs_config=gcs_config,
            logger=logger,
        )

    assert mock_read_parquet.call_count == 2
    assert len(df) == 2
    assert "Loaded 1 rows" in caplog.text