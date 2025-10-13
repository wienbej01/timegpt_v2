from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pandas as pd
import pytest

from timegpt_v2.config.model import ForecastExogConfig
from timegpt_v2.forecast.timegpt_client import (
    TimeGPTClient,
    TimeGPTConfig,
)


@pytest.fixture
def logger() -> logging.Logger:
    return logging.getLogger(__name__)


@pytest.mark.parametrize("strict_exog", [True, False])
def test_forecast_with_exogs_mocked_nixtla(
    strict_exog: bool, logger: logging.Logger
):
    """Test that the dataframes passed to Nixtla are correct."""
    mock_backend = MagicMock()

    def forecast_side_effect(df, X_df, *, hist_exog_list, **kwargs):
        # Assertions on the dataframes passed to Nixtla
        assert "unique_id" in df.columns
        assert "ds" in df.columns
        assert "y" in df.columns
        assert "hist_exog_1" in df.columns
        if strict_exog:
            assert df.empty
        else:
            assert "missing_hist" in df.columns

        if X_df is not None:
            assert "unique_id" in X_df.columns
            assert "ds" in X_df.columns
            assert "futr_exog_1" in X_df.columns
            assert "missing_futr" in X_df.columns

        return pd.DataFrame(
            {
                "unique_id": ["a"] * 3,
                "quantile": [0.25, 0.5, 0.75],
                "value": [0.9, 1.0, 1.1],
            }
        )

    mock_backend.forecast.side_effect = forecast_side_effect

    exog_config = ForecastExogConfig(
        use_exogs=True,
        strict_exog=strict_exog,
        hist_exog_list_raw=["hist_exog_1", "missing_hist"],
        futr_exog_list_raw=["futr_exog_1", "missing_futr"],
    )
    config = TimeGPTConfig(api_mode="online", exog=exog_config)
    mock_cache = MagicMock()
    mock_cache.get.return_value = None
    client = TimeGPTClient(backend=mock_backend, cache=mock_cache, config=config, logger=logger)

    y_df = pd.DataFrame({
        "unique_id": ["a"],
        "ds": [pd.to_datetime("2025-01-01", utc=True)],
        "y": [1],
    })
    x_df = pd.DataFrame({
        "unique_id": ["a"],
        "ds": [pd.to_datetime("2025-01-02", utc=True)],
    })
    features = pd.DataFrame({
        "timestamp": [
            pd.to_datetime("2025-01-01", utc=True),
            pd.to_datetime("2025-01-02", utc=True),
        ],
        "symbol": ["a", "a"],
        "hist_exog_1": [10, 11],
        "futr_exog_1": [100, 101],
    })

    client.forecast(
        y_df, x_df, features=features, snapshot_ts=pd.to_datetime("2025-01-01", utc=True), run_id="test"
    )
    mock_backend.forecast.assert_called_once()