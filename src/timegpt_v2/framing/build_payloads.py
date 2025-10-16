"""Payload construction utilities for TimeGPT requests."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from datetime import timedelta

import pandas as pd

from timegpt_v2.fe import deterministic
from timegpt_v2.fe.deterministic import (
    get_deterministic_exog_names,
    build_det_exogs_for_history,
    build_det_exogs_for_future,
)

# Historical exogenous features (appear only in y_df)
HIST_EXOG_ALLOW = [
    "ret_1m",
    "ret_5m",
    "sigma_5m",
    "parkinson_sigma_15m",
    "range_pct_15m",
    "clv_15m",
    "vwap_dev",
    "rth_cumret_30m",
    "spy_ret_1m",
    "spy_vol_30m",
    "regime_high_vol",
    "regime_high_dispersion",
]

# Future deterministic exogenous features (appear in both y_df and x_df)
# Single source of truth from deterministic module
FUTR_EXOG_ALLOW = get_deterministic_exog_names()


def build_y_df(
    features: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    *,
    target_column: str = "target_log_return_1m",
    rolling_window_days: int = 40,
    min_obs_subhourly: int = 1008,
    symbols: Iterable[str] | None = None,
    tz: str = "America/New_York",
    rth_open: str = "09:30",
    rth_close: str = "16:00",
) -> pd.DataFrame:
    """Return the ``y`` frame for TimeGPT up to *snapshot_ts* (inclusive) with rolling window."""
    if target_column not in features.columns:
        raise KeyError(f"Target column '{target_column}' missing from features")

    working = features.copy()
    if symbols is not None:
        working = working[working["symbol"].isin(symbols)]

    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True)
    cutoff = pd.to_datetime(snapshot_ts, utc=True)

    window_start = cutoff - pd.Timedelta(days=rolling_window_days)
    filtered = working.loc[working["timestamp"] <= cutoff].copy()
    filtered = filtered.loc[filtered["timestamp"] >= window_start].copy()
    filtered = filtered.loc[filtered[target_column].notna()]
    if filtered.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    # Enforce min_obs_subhourly for minute data
    if min_obs_subhourly > 0:
        counts = filtered.groupby("symbol").size()
        valid_symbols = counts[counts >= min_obs_subhourly].index
        filtered = filtered[filtered["symbol"].isin(valid_symbols)]

    if filtered.empty:
        return pd.DataFrame(columns=["unique_id", "ds", "y"])

    # Build base frame with unique_id, ds, y and historical exogs
    filtered.sort_values(["symbol", "timestamp"], inplace=True)
    renamed = filtered.rename(
        columns={"symbol": "unique_id", "timestamp": "ds", target_column: "y"}
    )

    # Fill gaps to make continuous time series for TimeGPT
    # Enforce minute-level regularity
    renamed = (
        renamed.set_index("ds")
        .groupby("unique_id")
        .resample("1min", include_groups=False)
        .asfreq()
        .reset_index()
    )
    # Forward fill y for non-trading hours to avoid NaN in target
    renamed["y"] = renamed.groupby("unique_id")["y"].ffill()

    # Log minute regularity enforcement
    if os.environ.get("PAYLOAD_LOG", "0") == "1":
        logger = logging.getLogger("timegpt_v2.build_payloads")
        unique_ids = renamed["unique_id"].nunique()
        total_rows = len(renamed)
        avg_rows_per_id = total_rows / unique_ids if unique_ids > 0 else 0
        logger.info(
            "Minute regularity enforced: %d symbols, avg %.1f rows per symbol (1min freq)",
            unique_ids, avg_rows_per_id
        )

    # Select historical exogenous features (HIST_EXOG_ALLOW)
    hist_exog_cols = [col for col in HIST_EXOG_ALLOW if col in renamed.columns]
    if hist_exog_cols:
        renamed[hist_exog_cols] = renamed.groupby("unique_id")[hist_exog_cols].ffill()

    # Compute deterministic exogs using new builder (no merges, avoids suffixes)
    base_bars_for_det = pd.DataFrame({
        "timestamp": renamed["ds"],
        "symbol": renamed["unique_id"]
    })
    det_exogs_history = build_det_exogs_for_history(
        base_bars_for_det,
        tz=tz,
        rth_open=rth_open,
        rth_close=rth_close
    )

    # Concatenate deterministic exogs column-wise (no merges, no suffixes)
    result_dfs = [renamed[["unique_id", "ds", "y"]]]
    if hist_exog_cols:
        result_dfs.append(renamed[hist_exog_cols])

    # Deterministic exogs from new builder with correct dtypes
    result_dfs.append(det_exogs_history)

    # Concatenate by index to avoid any suffix creation
    result = pd.concat(result_dfs, axis=1)

    # Ensure proper column order
    cols_to_select = ["unique_id", "ds", "y"]
    present_hist_cols = [col for col in HIST_EXOG_ALLOW if col in result.columns]
    cols_to_select.extend(present_hist_cols)
    cols_to_select.extend(FUTR_EXOG_ALLOW)  # Deterministic exogs
    result = result[cols_to_select].reset_index(drop=True)

    # Log payload columns if PAYLOAD_LOG is enabled
    if os.environ.get("PAYLOAD_LOG", "0") == "1":
        logger = logging.getLogger("timegpt_v2.build_payloads")
        logger.info(
            "PAYLOAD y_df cols=%s (hist_exog=%d, det_exog=%d)",
            str(list(result.columns)),
            len(present_hist_cols),
            len(FUTR_EXOG_ALLOW)
        )

    return result


def build_x_df_for_horizon(
    features: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    horizon_minutes: int,
    *,
    symbols: Iterable[str] | None = None,
    strict_exog: bool = False,
    tz: str = "America/New_York",
    rth_open: str = "09:30",
    rth_close: str = "16:00",
) -> pd.DataFrame:
    """Construct deterministic projections for minutes after *snapshot_ts*."""
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")

    base_symbols = list(symbols) if symbols is not None else []
    if not base_symbols:
        if "symbol" not in features.columns:
            raise KeyError("features must include a 'symbol' column when symbols not provided")
        base_symbols = sorted(str(sym) for sym in features["symbol"].unique())

    snapshot = pd.to_datetime(snapshot_ts, utc=True)
    future_timestamps = pd.date_range(
        snapshot + timedelta(minutes=1),
        periods=horizon_minutes,
        freq="1min",
    )

    # Build future frames using new deterministic builder (no merges, avoids suffixes)
    future_frames = []
    for symbol in base_symbols:
        # Create future index dataframe for this symbol
        future_df = pd.DataFrame({
            "timestamp": future_timestamps,
            "symbol": symbol
        })

        # Use new deterministic builder for future timestamps
        det_exogs_future = build_det_exogs_for_future(
            future_df,
            tz=tz,
            rth_open=rth_open,
            rth_close=rth_close
        )

        # Add unique_id and rename timestamp to ds
        det_exogs_future["unique_id"] = symbol
        det_exogs_future["ds"] = future_timestamps

        # Select only unique_id, ds and deterministic exogs
        cols_to_select = ["unique_id", "ds"] + FUTR_EXOG_ALLOW
        future_frames.append(det_exogs_future[cols_to_select])

    if not future_frames:
        return pd.DataFrame()

    # Concatenate by index to avoid any suffix creation
    result = pd.concat(future_frames, ignore_index=True)
    result.sort_values(["unique_id", "ds"], inplace=True)

    # Strict exog validation: ensure all FUTR_EXOG_ALLOW features are present
    if strict_exog:
        missing_futr_exog = [col for col in FUTR_EXOG_ALLOW if col not in result.columns]
        if missing_futr_exog:
            raise ValueError(
                f"strict_exog=True: missing deterministic exogenous features: {missing_futr_exog}. "
                f"These features are required for x_df but were not generated by build_det_exogs_for_future."
            )

    # Build final column list with only allowed deterministic features
    ordered_cols = ["unique_id", "ds"] + FUTR_EXOG_ALLOW
    result = result[ordered_cols].reset_index(drop=True)

    # Log payload columns if PAYLOAD_LOG is enabled
    if os.environ.get("PAYLOAD_LOG", "0") == "1":
        logger = logging.getLogger("timegpt_v2.build_payloads")
        logger.info(
            "PAYLOAD x_df cols=%s (det_exog=%d, strict_exog=%s)",
            str(list(result.columns)),
            len(FUTR_EXOG_ALLOW),
            strict_exog
        )

    return result


def _validate_horizon_length(x_df: pd.DataFrame, horizon_minutes: int) -> None:
    """Validate that X_df has exactly horizon rows per series.

    Args:
        x_df: Future DataFrame with deterministic exogenous features
        horizon_minutes: Expected number of future minutes

    Raises:
        ValueError: If X_df length doesn't match horizon
    """
    if x_df.empty:
        return

    length_check = x_df.groupby("unique_id").size()
    invalid_series = length_check[length_check != horizon_minutes]

    if not invalid_series.empty:
        raise ValueError(
            f"X_df horizon length mismatch: expected {horizon_minutes} rows per series, "
            f"found {invalid_series.to_dict()}. "
            f"This indicates a problem with future timestamp generation."
        )


def _validate_exog_parity(y_df: pd.DataFrame, x_df: pd.DataFrame) -> None:
    """Validate exogenous feature parity between historical (y_df) and future (x_df) DataFrames.

    This function ensures that:
    1. No _x/_y suffixes exist in any payload columns
    2. Deterministic exog names are identical in both DataFrames
    3. Deterministic exog dtypes match between DataFrames

    Args:
        y_df: Historical DataFrame with features and target
        x_df: Future DataFrame with deterministic exogenous features

    Raises:
        ValueError: If forbidden suffixes found or deterministic exog names mismatch
        TypeError: If deterministic exog dtypes mismatch between DataFrames
    """
    # Check for forbidden _x/_y suffixes
    all_columns = list(y_df.columns) + list(x_df.columns)
    forbidden_suffixes = [col for col in all_columns if col.endswith("_x") or col.endswith("_y")]
    if forbidden_suffixes:
        raise ValueError(f"Forbidden suffixes in payload columns: {sorted(set(forbidden_suffixes))}")

    # Check deterministic exog name parity
    det_exog_names = set(get_deterministic_exog_names())
    y_det_cols = det_exog_names & set(y_df.columns)
    x_det_cols = det_exog_names & set(x_df.columns)

    if y_det_cols != x_det_cols:
        y_only = sorted(y_det_cols - x_det_cols)
        x_only = sorted(x_det_cols - y_det_cols)
        raise ValueError(
            f"Deterministic exog name mismatch: in_y_only={y_only}, in_x_only={x_only}"
        )

    # Check deterministic exog dtype parity
    dtype_mismatches = []
    for col in sorted(det_exog_names):
        if col in y_df.columns and col in x_df.columns:
            y_dtype = str(y_df[col].dtype)
            x_dtype = str(x_df[col].dtype)
            if y_dtype != x_dtype:
                dtype_mismatches.append((col, y_dtype, x_dtype))

    if dtype_mismatches:
        raise TypeError(f"Deterministic exog dtype mismatch: {dtype_mismatches}")


def build_payloads_with_validation(
    features: pd.DataFrame,
    snapshot_ts: pd.Timestamp,
    horizon_minutes: int,
    *,
    target_column: str = "target_log_return_1m",
    rolling_window_days: int = 40,
    min_obs_subhourly: int = 1008,
    symbols: Iterable[str] | None = None,
    strict_exog: bool = False,
    tz: str = "America/New_York",
    rth_open: str = "09:30",
    rth_close: str = "16:00",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build validated TimeGPT payloads with deterministic exog parity.

    This function builds both y_df (historical) and x_df (future) DataFrames
    and validates exogenous feature parity before returning them to the caller.

    Args:
        features: Historical feature matrix with target and exogenous columns
        snapshot_ts: Snapshot timestamp for forecast
        horizon_minutes: Number of future minutes to forecast
        target_column: Name of target column in features
        rolling_window_days: Rolling window size for historical data
        min_obs_subhourly: Minimum observations required per symbol
        symbols: List of symbols to include (None for all)
        strict_exog: Whether to enforce strict exog presence
        tz: Timezone for deterministic features
        rth_open: Regular trading hours open time
        rth_close: Regular trading hours close time

    Returns:
        Tuple of (y_df, x_df) with validated exogenous feature parity

    Raises:
        ValueError: If forbidden suffixes found or exog names mismatch
        TypeError: If exog dtypes mismatch between DataFrames
    """
    # Build y_df with deterministic exogs
    y_df = build_y_df(
        features=features,
        snapshot_ts=snapshot_ts,
        target_column=target_column,
        rolling_window_days=rolling_window_days,
        min_obs_subhourly=min_obs_subhourly,
        symbols=symbols,
        tz=tz,
        rth_open=rth_open,
        rth_close=rth_close,
    )

    # Build x_df with deterministic exogs only
    x_df = build_x_df_for_horizon(
        features=features,
        snapshot_ts=snapshot_ts,
        horizon_minutes=horizon_minutes,
        symbols=symbols,
        strict_exog=strict_exog,
        tz=tz,
        rth_open=rth_open,
        rth_close=rth_close,
    )

    # Validate horizon length and exogenous feature parity before returning
    # Only validate if both DataFrames are non-empty
    if not y_df.empty and not x_df.empty:
        _validate_horizon_length(x_df, horizon_minutes)
        _validate_exog_parity(y_df, x_df)
    elif not y_df.empty:
        # y_df has data but x_df doesn't - this shouldn't happen
        raise ValueError("y_df has data but x_df is empty - this indicates a payload construction error")
    elif not x_df.empty:
        # x_df has data but y_df doesn't - this can happen with insufficient history
        # Still validate horizon length
        _validate_horizon_length(x_df, horizon_minutes)

    return y_df, x_df


__all__ = ["build_y_df", "build_x_df_for_horizon", "build_payloads_with_validation"]
