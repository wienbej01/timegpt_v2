"""Tests for batcher."""

import pandas as pd

from timegpt_v2.forecast.batcher import build_batches
from timegpt_v2.utils.payload import estimate_bytes


def test_build_batches():
    """Test building batches."""
    y_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b", "c", "c"],
            "ds": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02", "2025-01-01", "2025-01-02"]),
            "y": [1, 2, 3, 4, 5, 6],
        }
    )
    x_df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b", "c", "c"],
            "ds": pd.to_datetime(["2025-01-03", "2025-01-04", "2025-01-03", "2025-01-04", "2025-01-03", "2025-01-04"]),
            "x": [1, 2, 3, 4, 5, 6],
        }
    )

    # Test case for 2 batches
    max_bytes = estimate_bytes(y_df[y_df["unique_id"].isin(["a", "b"])]) + estimate_bytes(x_df[x_df["unique_id"].isin(["a", "b"])]) + 1
    batches = list(build_batches(y_df, x_df, "unique_id", max_bytes))
    assert len(batches) == 2
    assert set(batches[0][2]["unique_ids"]) == {"a", "b"}
    assert batches[1][2]["unique_ids"] == ["c"]

    # Test case for 3 batches
    max_bytes = estimate_bytes(y_df[y_df["unique_id"] == "a"]) + estimate_bytes(x_df[x_df["unique_id"] == "a"]) + 1
    batches = list(build_batches(y_df, x_df, "unique_id", max_bytes))
    assert len(batches) == 3
    assert batches[0][2]["unique_ids"] == ["a"]
    assert batches[1][2]["unique_ids"] == ["b"]
    assert batches[2][2]["unique_ids"] == ["c"]

    # Test case for 1 batch
    max_bytes = estimate_bytes(y_df) + estimate_bytes(x_df) + 1
    batches = list(build_batches(y_df, x_df, "unique_id", max_bytes))
    assert len(batches) == 1
    assert set(batches[0][2]["unique_ids"]) == {"a", "b", "c"}
