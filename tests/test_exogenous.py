from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from timegpt_v2.forecast.exogenous import (
    build_future_frame,
    estimate_payload_bytes,
    merge_history_exogs,
    normalize_names,
    select_available,
)


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger(__name__)


@pytest.fixture
def y_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AAPL"] * 10,
            "ds": pd.to_datetime(pd.date_range("2024-01-01 09:30", periods=10, freq="1min"), utc=True),
            "y": np.arange(10, dtype=float),
        }
    )


@pytest.fixture
def features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.date_range("2024-01-01 09:30", periods=10, freq="1min"), utc=True),
            "symbol": ["AAPL"] * 10,
            "hist_feat_1": np.arange(10, dtype=float),
            "hist_feat_2": np.arange(10, dtype=float) * 2,
        }
    )


@pytest.fixture
def x_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["AAPL"] * 5,
            "ds": pd.to_datetime(pd.date_range("2024-01-01 09:40", periods=5, freq="1min"), utc=True),
        }
    )


@pytest.fixture
def future_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pd.date_range("2024-01-01 09:40", periods=5, freq="1min"), utc=True),
            "symbol": ["AAPL"] * 5,
            "futr_feat_1": np.arange(5, dtype=float),
        }
    )


def test_merge_history_exogs_present_all(logger, y_df, features_df):
    exogs = ["hist_feat_1", "hist_feat_2"]
    result = merge_history_exogs(y_df, features_df, exogs, False, "none", logger)
    expected = y_df.copy()
    expected["hist_feat_1"] = np.arange(10, dtype=float)
    expected["hist_feat_2"] = np.arange(10, dtype=float) * 2
    assert_frame_equal(result, expected)


def test_merge_history_exogs_missing_permissive(logger, y_df, features_df):
    exogs = ["hist_feat_1", "missing_feat"]
    result = merge_history_exogs(y_df, features_df, exogs, False, "none", logger)
    expected = y_df.copy()
    expected["hist_feat_1"] = np.arange(10, dtype=float)
    assert_frame_equal(result, expected)


def test_merge_history_exogs_missing_strict(logger, y_df, features_df):
    exogs = ["missing_feat"]
    with pytest.raises(ValueError, match="Missing historical exogenous features"):
        merge_history_exogs(y_df, features_df, exogs, True, "none", logger)


def test_build_future_frame_present_all(logger, x_df, future_features_df):
    exogs = ["futr_feat_1"]
    result = build_future_frame(x_df, future_features_df, exogs, False, logger)
    expected = x_df.copy()
    expected["futr_feat_1"] = np.arange(5, dtype=float)
    assert_frame_equal(result, expected)


def test_build_future_frame_missing_permissive(logger, x_df, future_features_df):
    exogs = ["futr_feat_1", "missing_feat"]
    result = build_future_frame(x_df, future_features_df, exogs, False, logger)
    expected = x_df.copy()
    expected["futr_feat_1"] = np.arange(5, dtype=float)
    assert_frame_equal(result, expected)


def test_normalize_names_and_select_available():
    names = ["a", "b", "c", "a"]
    name_map = {"a": "A", "c": "C"}
    normalized = normalize_names(names, name_map)
    assert normalized == ["A", "b", "C"]

    columns = ["A", "C", "d"]
    present, missing = select_available(normalized, columns)
    assert present == ["A", "C"]
    assert missing == ["b"]


def test_estimate_payload_bytes():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert estimate_payload_bytes(df) > 0