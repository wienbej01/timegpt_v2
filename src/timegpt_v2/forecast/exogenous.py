
# src/timegpt_v2/forecast/exogenous.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger("timegpt_v2.forecast")

def normalize_names(names: List[str], name_map: dict[str, str]) -> List[str]:
    return [name_map.get(name, name) for name in names]

def select_available(declared: List[str], available: pd.Index) -> Tuple[List[str], List[str]]:
    present = [col for col in declared if col in available]
    missing = [col for col in declared if col not in available]
    return present, missing

def preflight_log(
    lg: logging.Logger,
    hist_exog_declared: List[str],
    hist_exog_present: List[str],
    hist_exog_missing: List[str],
    futr_exog_declared: List[str],
    futr_exog_present: List[str],
    futr_exog_missing: List[str],
    y_shape_before: Tuple[int, int],
    y_shape_after: Tuple[int, int],
    x_shape_before: Tuple[int, int],
    x_shape_after: Tuple[int, int],
) -> None:
    lg.info("--- Exogenous Features Preflight ---")
    lg.info("History exogs declared: %s", hist_exog_declared)
    lg.info("History exogs present: %s", hist_exog_present)
    if hist_exog_missing:
        lg.warning("History exogs missing: %s", hist_exog_missing)
    lg.info("Future exogs declared: %s", futr_exog_declared)
    lg.info("Future exogs present: %s", futr_exog_present)
    if futr_exog_missing:
        lg.warning("Future exogs missing: %s", futr_exog_missing)
    lg.info("y_df shape before/after: %s -> %s", y_shape_before, y_shape_after)
    lg.info("x_df shape before/after: %s -> %s", x_shape_before, x_shape_after)

def estimate_payload_bytes(df: pd.DataFrame) -> int:
    if df is None:
        return 0
    return df.memory_usage(deep=True).sum()

def build_future_frame(
    x_df: pd.DataFrame,
    features_df: pd.DataFrame,
    futr_exogs: list[str],
    strict: bool,
    log: logging.Logger,
) -> pd.DataFrame:
    # This function is being replaced by merge_future_exogs
    if x_df is None:
        return pd.DataFrame() # Should not happen if futr_exogs is not empty
    return x_df

def merge_history_exogs(
    y_df: pd.DataFrame,
    features_df: pd.DataFrame,
    exogs: List[str],
    strict: bool = True,
    impute_strategy: str = "none",
    log: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    features_df = features_df.copy()
    if "timestamp" in features_df.columns:
        features_df.rename(columns={"timestamp": "ds"}, inplace=True)
    if "symbol" in features_df.columns:
        features_df.rename(columns={"symbol": "unique_id"}, inplace=True)
    """
    Merge historical exogenous variables into y_df and return a dataframe
    that contains the bare exog names without pandas suffixes.
    - If y_df already contains some of exogs, we coalesce with features_df values.
    - If impute_strategy == 'zero', impute numerics with 0 and booleans with False.
    - If strict is True, drop rows with NaNs in required exogs (after coalesce/impute).
    """
    lg = log or logger

    # Sanity: keys must exist
    for col in ("unique_id", "ds"):
        if col not in y_df.columns:
            raise KeyError(f"y_df missing required column '{col}'")
        if col not in features_df.columns:
            raise KeyError(f"features_df missing required column '{col}'")

    # Select only the columns we need from features to avoid accidental suffix explosion
    needed_cols = ["unique_id", "ds"] + [c for c in exogs if c in features_df.columns]
    feats = features_df[needed_cols].copy()

    # Merge with controlled suffixes to avoid _x/_y ambiguity
    merged = y_df.merge(
        feats,
        on=["unique_id", "ds"],
        how="left",
        suffixes=("", "_feat"),
    )

    # Coalesce any existing columns in y_df with the _feat columns from features_df
    for c in exogs:
        base = c
        feat = f"{c}_feat"

        if base in merged.columns and feat in merged.columns:
            # y_df value wins unless NaN, then take features
            merged[base] = merged[base].where(~merged[base].isna(), merged[feat])
            merged.drop(columns=[feat], inplace=True)
        elif feat in merged.columns and base not in merged.columns:
            # Only features had it; rename to bare name
            merged.rename(columns={feat: base}, inplace=True)
        elif base not in merged.columns:
            # Neither side had it; create it as NaN to allow downstream handling
            merged[base] = np.nan

    # Optional imputation
    if impute_strategy == "zero":
        for c in exogs:
            if c not in merged.columns:
                merged[c] = np.nan
            # Boolean or numeric fill
            if pd.api.types.is_bool_dtype(merged[c]):
                merged[c] = merged[c].fillna(False)
            else:
                merged[c] = merged[c].fillna(0.0)

    # Strict enforcement: drop rows with NaNs in bare exog names
    if strict:
        missing = [c for c in exogs if c not in merged.columns]
        if missing:
            raise KeyError(f"Missing required exogs after merge: {missing}")
        rows_before = len(merged)
        merged.dropna(subset=exogs, inplace=True)
        rows_after = len(merged)
        if rows_after < rows_before:
            lg.warning(f"Dropped {rows_before - rows_after} rows with NaNs in hist exogs {exogs}")

    return merged

def merge_future_exogs(
    x_df: pd.DataFrame,
    features_df: pd.DataFrame,
    futr_exogs: List[str],
    strict: bool = True,
) -> pd.DataFrame:
    features_df = features_df.copy()
    if "timestamp" in features_df.columns:
        features_df.rename(columns={"timestamp": "ds"}, inplace=True)
    if "symbol" in features_df.columns:
        features_df.rename(columns={"symbol": "unique_id"}, inplace=True)
    """
    Left-join future exogs onto the future frame (x_df) on ['unique_id','ds'].
    For boolean event flags, default missing to False when strict=True.
    Uses controlled suffixes to avoid _x/_y ambiguity like merge_history_exogs.
    """
    if not futr_exogs:
        return x_df

    needed_cols = ["unique_id", "ds"] + [c for c in futr_exogs if c in features_df.columns]
    feats = features_df[needed_cols].copy()

    # Merge with controlled suffixes to avoid _x/_y ambiguity (same as merge_history_exogs)
    out = x_df.merge(
        feats,
        on=["unique_id", "ds"],
        how="left",
        suffixes=("", "_feat"),
    )

    # Coalesce any existing columns in x_df with the _feat columns from features_df
    for c in futr_exogs:
        base = c
        feat = f"{c}_feat"

        if base in out.columns and feat in out.columns:
            # x_df value wins unless NaN, then take features
            out[base] = out[base].where(~out[base].isna(), out[feat])
            out.drop(columns=[feat], inplace=True)
        elif feat in out.columns and base not in out.columns:
            # Only features had it; rename to bare name
            out.rename(columns={feat: base}, inplace=True)
        elif base not in out.columns:
            # Neither side had it; create it as NaN to allow downstream handling
            out[base] = np.nan

    if strict:
        for c in futr_exogs:
            if c not in out.columns:
                # Create and default to False for event flags
                out[c] = False
            elif pd.api.types.is_bool_dtype(out[c]):
                out[c] = out[c].fillna(False)

    return out
