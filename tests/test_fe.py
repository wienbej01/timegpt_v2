from __future__ import annotations

import pandas as pd

from timegpt_v2.fe import deterministic
from timegpt_v2.fe.base_features import build_feature_matrix
from timegpt_v2.fe.context import FeatureContext
from timegpt_v2.utils.synthetic import SyntheticConfig, generate_bars


def _build_multisymbol_dataframe() -> pd.DataFrame:
    configs = [
        SyntheticConfig(symbol="AAPL", sessions=2, minutes_per_session=390),
        SyntheticConfig(symbol="MSFT", sessions=2, minutes_per_session=390, seed=11),
        SyntheticConfig(symbol="SPY", sessions=2, minutes_per_session=390, seed=17),
    ]
    frames = [generate_bars(cfg) for cfg in configs]
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    return combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def test_feature_pipeline_no_future_leakage() -> None:
    raw = _build_multisymbol_dataframe()
    events = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02 10:00", tz="America/New_York")],
            "event_type": ["earnings"],
        }
    )
    context = FeatureContext(
        symbols=["AAPL", "MSFT"], events=events, market_symbol="SPY", market_data=raw
    )
    features = build_feature_matrix(raw, context=context)

    required_columns = {
        "ret_1m",
        "ret_5m",
        "ret_15m",
        "ret_30m",
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
        "minute_index",
        "fourier_sin_1",
        "session_open",
        "spy_ret_1m",
        "regime_high_vol",
        "event_earnings",
    }
    assert required_columns.issubset(features.columns)

    assert {"spy_ret_1m", "regime_high_vol", "event_earnings"}.issubset(features.columns)

    for symbol in ["AAPL", "MSFT"]:
        feature_slice = features[features["symbol"] == symbol]
        assert (feature_slice["label_timestamp"] >= feature_slice["timestamp"]).all()
        assert feature_slice["target_log_return_1m"].notna().all()


def test_feature_nan_rate_below_threshold() -> None:
    raw = _build_multisymbol_dataframe()
    features = build_feature_matrix(
        raw,
        context=FeatureContext(symbols=["AAPL", "MSFT", "SPY"], market_data=raw),
    )
    feature_columns = [
        col for col in features.columns if col not in {"timestamp", "symbol", "label_timestamp"}
    ]
    nan_ratio = features[feature_columns].isna().mean().mean()
    assert nan_ratio < 0.01


def test_deterministic_features_reproducible() -> None:
    raw = _build_multisymbol_dataframe().head(50)[["timestamp", "symbol"]]
    first = deterministic.add_time_features(raw)
    second = deterministic.add_time_features(raw)
    pd.testing.assert_frame_equal(first, second)
