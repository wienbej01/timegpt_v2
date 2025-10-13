"""Centralized schema for all dataframes."""

from __future__ import annotations

# Base columns from the raw data
RAW_DATA_COLS = frozenset(
    ["timestamp", "open", "high", "low", "close", "volume", "vw", "n", "session", "date_et", "symbol"]
)

# Columns added by the DataQualityChecker
DQ_CHECKER_COLS = frozenset(["ffill_flag", "log_return", "outlier_flag", "is_rth"])

# All columns in the clean dataframe
CLEAN_DF_COLS = RAW_DATA_COLS | DQ_CHECKER_COLS

# Target columns
TARGET_COLS = frozenset(
    [
        "target_log_return_1m",
        "target_log_return_15m",
        "target_bp_ret_1m",
        "target_z_ret_1m",
    ]
)

# Exogenous feature columns
HISTORICAL_EXOG_COLS = frozenset(
    [
        "spy_ret_1m",
        "spy_vol_30m",
        "regime_high_vol",
        "regime_high_dispersion",
    ]
)

FUTURE_EXOG_COLS = frozenset(["event_earnings", "event_fomc", "event_cpi"])

ALL_EXOG_COLS = HISTORICAL_EXOG_COLS | FUTURE_EXOG_COLS

# All columns in the feature matrix
FEATURE_MATRIX_COLS = (
    CLEAN_DF_COLS
    | TARGET_COLS
    | ALL_EXOG_COLS
    | frozenset(
        [
            "ret_1m",
            "ret_5m",
            "ret_15m",
            "ret_30m",
            "vol_ewm_60m",
            "vol_ewm_15m",
            "rv_5m",
            "rv_15m",
            "rv_30m",
            "ret_skew_15m",
            "ret_kurt_15m",
            "atr_5m",
            "vol_parkinson_30m",
            "vol_garman_klass_30m",
            "vwap_30m",
            "z_close_vwap_30m",
            "vwap_trend_5m",
            "vol_5m_norm",
            "volume_percentile_20d",
            "range_pct",
            "signed_volume_5m",
            "label_timestamp",
            "label_timestamp_15m",
            "minute_of_day",
            "minute_index",
            "minute_progress",
            "fourier_sin_1",
            "fourier_cos_1",
            "fourier_sin_2",
            "fourier_cos_2",
            "fourier_sin_3",
            "fourier_cos_3",
            "session_open",
            "session_lunch",
            "session_power",
            "day_of_week",
            "is_month_end",
            "spy_ret_1m",
            "spy_vol_30m",
            "dispersion_z",
            "regime_high_dispersion",
            "regime_high_vol",
            "event_earnings",
            "event_fomc",
            "event_cpi",
            "timestamp_et",
        ]
    )
)
