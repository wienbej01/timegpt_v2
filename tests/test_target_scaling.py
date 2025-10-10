from __future__ import annotations

import pandas as pd
import pytest

from timegpt_v2.forecast.scaling import TargetScaler, TargetScalingConfig


@pytest.fixture
def sample_features() -> pd.DataFrame:
    data = {
        "symbol": ["AAPL", "AAPL"],
        "label_timestamp": pd.to_datetime(
            ["2024-07-01T14:15:00Z", "2024-07-01T14:30:00Z"], utc=True
        ),
        "label_timestamp_15m": pd.to_datetime(
            ["2024-07-01T14:30:00Z", "2024-07-01T14:45:00Z"], utc=True
        ),
        "target_log_return_1m": [0.001, -0.0005],
        "target_log_return_15m": [0.015, -0.007],
        "target_bp_ret_1m": [10.0, -5.0],
        "target_z_ret_1m": [0.2, -0.1],
        "vol_ewm_60m": [0.005, 0.005],
    }
    return pd.DataFrame(data)


def test_basis_point_scaling_inverse(sample_features: pd.DataFrame) -> None:
    config = TargetScalingConfig(mode="basis_point", bp_factor=10_000.0)
    scaler = TargetScaler(config)
    forecasts = pd.DataFrame(
        {
            "unique_id": ["AAPL"],
            "forecast_ts": pd.to_datetime(["2024-07-01T14:15:00Z"], utc=True),
            "q25": [250.0],
            "q50": [0.0],
            "q75": [750.0],
        }
    )
    result = scaler.inverse_quantiles(
        forecasts, features=sample_features, quantile_columns=["q25", "q50", "q75"]
    )
    assert pytest.approx(result.loc[0, "q25"], rel=1e-9) == 0.025
    assert pytest.approx(result.loc[0, "q50"], rel=1e-9) == 0.0
    assert pytest.approx(result.loc[0, "q75"], rel=1e-9) == 0.075


def test_volatility_scaling_inverse(sample_features: pd.DataFrame) -> None:
    config = TargetScalingConfig(mode="volatility_z", volatility_column="vol_ewm_60m")
    scaler = TargetScaler(config)
    forecasts = pd.DataFrame(
        {
            "unique_id": ["AAPL"],
            "forecast_ts": pd.to_datetime(["2024-07-01T14:30:00Z"], utc=True),
            "q25": [-1.0],
            "q50": [0.0],
            "q75": [1.0],
        }
    )
    result = scaler.inverse_quantiles(
        forecasts, features=sample_features, quantile_columns=["q25", "q50", "q75"]
    )
    assert pytest.approx(result.loc[0, "q25"], rel=1e-9) == -0.005
    assert pytest.approx(result.loc[0, "q75"], rel=1e-9) == 0.005


def test_scaler_metadata_modes() -> None:
    config = TargetScalingConfig(mode="log_return")
    scaler = TargetScaler(config)
    metadata = scaler.metadata
    assert metadata["mode"] == "log_return"


def test_scaler_log_return_15m_columns(sample_features: pd.DataFrame) -> None:
    config = TargetScalingConfig(mode="log_return_15m")
    scaler = TargetScaler(config)
    assert scaler.target_column == "target_log_return_15m"
    assert scaler.label_timestamp_column == "label_timestamp_15m"
