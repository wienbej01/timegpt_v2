"""Payload estimation and splitting utilities."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def estimate_bytes(df: pd.DataFrame | None) -> int:
    """Estimate the memory usage of a DataFrame in bytes with a safety margin."""
    if df is None or df.empty:
        return 0
    return int(df.memory_usage(index=True, deep=True).sum() * 1.05)  # 5% margin


def split_by_ids(
    df: pd.DataFrame,
    id_col: str,
    max_bytes: int,
    min_ids: int = 1,
) -> Iterable[list[str]]:
    """Yield slices of unique IDs whose concatenated frames are under max_bytes."""
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in DataFrame.")

    ids = df[id_col].unique().tolist()
    batch_ids: list[str] = []
    current_bytes = 0

    for unique_id in ids:
        id_frame = df[df[id_col] == unique_id]
        id_bytes = estimate_bytes(id_frame)

        if batch_ids and (current_bytes + id_bytes > max_bytes):
            yield batch_ids
            batch_ids, current_bytes = [], 0

        batch_ids.append(unique_id)
        current_bytes += id_bytes

    if batch_ids:
        yield batch_ids
