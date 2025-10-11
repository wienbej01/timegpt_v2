"""Tests for payload utilities."""

import pandas as pd
import pytest

from timegpt_v2.utils.payload import estimate_bytes, split_by_ids


def test_estimate_bytes():
    """Test payload byte estimation."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert estimate_bytes(df) > 0
    assert estimate_bytes(None) == 0
    assert estimate_bytes(pd.DataFrame()) == 0


def test_split_by_ids():
    """Test splitting a DataFrame by IDs based on byte size."""
    df = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b", "c", "c"],
            "data": [1, 2, 3, 4, 5, 6],
        }
    )
    max_bytes = estimate_bytes(df[df["unique_id"] == "a"]) + 1

    chunks = list(split_by_ids(df, "unique_id", max_bytes))
    assert len(chunks) == 3
    assert chunks[0] == ["a"]
    assert chunks[1] == ["b"]
    assert chunks[2] == ["c"]

    max_bytes = estimate_bytes(df) + 1
    chunks = list(split_by_ids(df, "unique_id", max_bytes))
    assert len(chunks) == 1
    assert set(chunks[0]) == {"a", "b", "c"}
