from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def normalize_names(names: List[str], name_map: Dict[str, str]) -> List[str]:
    """Map names using name_map, preserving order and removing duplicates."""
    seen = set()
    normalized = []
    for name in names:
        mapped = name_map.get(name, name)
        if mapped not in seen:
            seen.add(mapped)
            normalized.append(mapped)
    return normalized


def select_available(
    declared: List[str], columns: Iterable[str]
) -> Tuple[List[str], List[str]]:
    """Return (present, missing) where present is a subset of declared."""
    present = [col for col in declared if col in columns]
    missing = [col for col in declared if col not in columns]
    return present, missing


def merge_history_exogs(
    y_df: pd.DataFrame,
    features_df: pd.DataFrame,
    exogs: List[str],
    strict: bool,
    impute_strategy: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Merge historical exogenous features into the history dataframe."""
    if not exogs:
        return y_df

    present_exogs, missing_exogs = select_available(exogs, features_df.columns)

    if missing_exogs:
        msg = f"Missing historical exogenous features: {missing_exogs}"
        if strict:
            logger.error(msg)
            raise ValueError(msg)
        logger.warning(f"Permissive mode. Dropping missing hist exogs: {missing_exogs}")

    if not present_exogs:
        return y_df

    features_to_merge = features_df.rename(columns={"symbol": "unique_id", "timestamp": "ds"})
    features_to_merge["ds"] = pd.to_datetime(features_to_merge["ds"], utc=True)

    y_df_exog = pd.merge(
        y_df,
        features_to_merge[["unique_id", "ds", *present_exogs]],
        on=["unique_id", "ds"],
        how="left",
    )

    if impute_strategy == "ffill":
        y_df_exog[present_exogs] = y_df_exog.groupby("unique_id")[present_exogs].ffill()
    elif impute_strategy == "bfill":
        y_df_exog[present_exogs] = y_df_exog.groupby("unique_id")[present_exogs].bfill()
    elif impute_strategy == "zero":
        y_df_exog[present_exogs] = y_df_exog[present_exogs].fillna(0)

    # Drop rows that still contain NaNs in required exogs
    rows_before = len(y_df_exog)
    y_df_exog.dropna(subset=present_exogs, inplace=True)
    rows_after = len(y_df_exog)
    if rows_before > rows_after:
        logger.warning(f"Dropped {rows_before - rows_after} rows with NaNs in hist exogs.")

    return y_df_exog


def build_future_frame(
    x_df: pd.DataFrame | None,
    future_features_df: pd.DataFrame,
    futr_exogs: List[str],
    strict: bool,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """Merge future exogenous features into the future dataframe."""
    if x_df is None or not futr_exogs:
        return x_df

    present_exogs, missing_exogs = select_available(futr_exogs, future_features_df.columns)

    if missing_exogs:
        msg = f"Missing future exogenous features: {missing_exogs}"
        if strict:
            logger.error(msg)
            raise ValueError(msg)
        logger.warning(f"Permissive mode. Dropping missing futr exogs: {missing_exogs}")

    if not present_exogs:
        return x_df

    features_to_merge = future_features_df.rename(
        columns={"symbol": "unique_id", "timestamp": "ds"}
    )
    features_to_merge["ds"] = pd.to_datetime(features_to_merge["ds"], utc=True)

    x_df_exog = pd.merge(
        x_df,
        features_to_merge[["unique_id", "ds", *present_exogs]],
        on=["unique_id", "ds"],
        how="left",
    )

    return x_df_exog


def preflight_log(
    logger: logging.Logger,
    declared_hist: List[str],
    present_hist: List[str],
    missing_hist: List[str],
    declared_futr: List[str],
    present_futr: List[str],
    missing_futr: List[str],
    y_shape_before: Tuple[int, int],
    y_shape_after: Tuple[int, int],
    x_shape_before: Tuple[int, int],
    x_shape_after: Tuple[int, int],
) -> None:
    """Emit concise structured logs for transparency."""
    logger.info("--- Exogenous Feature Preflight ---")
    logger.info(f"Declared hist exogs: {declared_hist}")
    logger.info(f"Present hist exogs:  {present_hist}")
    logger.info(f"Missing hist exogs:  {missing_hist}")
    logger.info(f"Declared futr exogs: {declared_futr}")
    logger.info(f"Present futr exogs:  {present_futr}")
    logger.info(f"Missing futr exogs:  {missing_futr}")
    logger.info(f"History shape before: {y_shape_before}, after: {y_shape_after}")
    logger.info(f"Future shape before:  {x_shape_before}, after: {x_shape_after}")
    logger.info("------------------------------------")


def estimate_payload_bytes(df: pd.DataFrame | None) -> int:
    """Rough estimate to log before call."""
    if df is None:
        return 0
    return df.memory_usage(deep=True).sum()
