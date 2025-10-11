"""Functions for creating batches of data for the TimeGPT API."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from timegpt_v2.utils.payload import estimate_bytes


def build_batches(
    y_df: pd.DataFrame,
    x_df: pd.DataFrame | None,
    id_col: str,
    max_bytes_per_call: int,
) -> Iterable[tuple[pd.DataFrame, pd.DataFrame | None, dict[str, object]]]:
    """Build batches of data for the TimeGPT API."""
    if y_df.empty:
        return

    all_ids = y_df[id_col].unique().tolist()
    batch_ids: list[str] = []
    current_bytes = 0

    for unique_id in all_ids:
        y_id_frame = y_df[y_df[id_col] == unique_id]
        x_id_frame = x_df[x_df[id_col] == unique_id] if x_df is not None else None
        id_bytes = estimate_bytes(y_id_frame) + estimate_bytes(x_id_frame)

        if batch_ids and (current_bytes + id_bytes > max_bytes_per_call):
            y_batch = y_df[y_df[id_col].isin(batch_ids)]
            x_batch = x_df[x_df[id_col].isin(batch_ids)] if x_df is not None else None
            batch_meta = {
                "unique_ids": batch_ids,
                "estimated_bytes": current_bytes,
            }
            yield y_batch, x_batch, batch_meta
            batch_ids, current_bytes = [], 0

        batch_ids.append(unique_id)
        current_bytes += id_bytes

    if batch_ids:
        y_batch = y_df[y_df[id_col].isin(batch_ids)]
        x_batch = x_df[x_df[id_col].isin(batch_ids)] if x_df is not None else None
        batch_meta = {
            "unique_ids": batch_ids,
            "estimated_bytes": current_bytes,
        }
        yield y_batch, x_batch, batch_meta
