import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from timegpt_v2.config.model import ForecastExogConfig
from timegpt_v2.forecast.timegpt_client import (
    PayloadTooLargeError,
    TimeGPTClient,
    TimeGPTConfig,
    _LocalDeterministicBackend,
)
from timegpt_v2.utils.api_budget import APIBudget
from timegpt_v2.utils.cache import ForecastCache


@pytest.fixture
def mock_backend():
    """Mock TimeGPT backend."""
    return MagicMock()


@pytest.fixture
def mock_cache():
    """Mock forecast cache."""
    return MagicMock()


@pytest.fixture
def logger():
    """Logger."""
    return logging.getLogger(__name__)


def test_offline_mode_uses_cache(mock_backend, mock_cache, logger):
    """Test that offline mode uses cache and makes no API calls."""
    config = TimeGPTConfig(api_mode="offline")
    client = TimeGPTClient(backend=mock_backend, cache=mock_cache, config=config, logger=logger)

    y_df = pd.DataFrame({"unique_id": ["a"], "ds": [pd.to_datetime("2025-01-01")], "y": [1]})
    mock_cache.get.return_value = {
        "values": {"0.25": 0.9, "0.5": 1.0, "0.75": 1.1},
        "forecast_ts": "2025-01-01T00:15:00Z",
        "snapshot_utc": "2025-01-01T00:00:00Z",
    }

    client.forecast(y_df, None, features=pd.DataFrame(), snapshot_ts=pd.to_datetime("2025-01-01"), run_id="test")

    mock_backend.forecast.assert_not_called()


def test_online_mode_calls_api_on_cache_miss(mock_backend, mock_cache, logger):
    """Test that online mode calls API on cache miss."""
    config = TimeGPTConfig(api_mode="online")
    client = TimeGPTClient(backend=mock_backend, cache=mock_cache, config=config, logger=logger)

    y_df = pd.DataFrame({"unique_id": ["a"], "ds": [pd.to_datetime("2025-01-01")], "y": [1]})
    mock_cache.get.return_value = None
    mock_backend.forecast.return_value = pd.DataFrame(
        {"unique_id": ["a"] * 3, "quantile": [0.25, 0.5, 0.75], "value": [0.9, 1.0, 1.1]}
    )

    client.forecast(y_df, None, features=pd.DataFrame(), snapshot_ts=pd.to_datetime("2025-01-01"), run_id="test")

    mock_backend.forecast.assert_called_once()
    mock_cache.put.assert_called_once()


def test_retry_on_payload_too_large(mock_backend, mock_cache, logger):
    """Test retry logic on PayloadTooLargeError."""
    config = TimeGPTConfig(api_mode="online")
    client = TimeGPTClient(backend=mock_backend, cache=mock_cache, config=config, logger=logger)

    y_df = pd.DataFrame({"unique_id": ["a", "b"], "ds": [pd.to_datetime("2025-01-01")] * 2, "y": [1, 2]})
    mock_cache.get.return_value = None

    def forecast_side_effect(*args, **kwargs):
        y_df_arg = args[0]
        if len(y_df_arg['unique_id'].unique()) > 1:
            raise PayloadTooLargeError
        else:
            unique_id = y_df_arg['unique_id'].unique()[0]
            if unique_id == 'a':
                return pd.DataFrame({"unique_id": ["a"] * 3, "quantile": [0.25, 0.5, 0.75], "value": [0.9, 1.0, 1.1]})
            elif unique_id == 'b':
                return pd.DataFrame({"unique_id": ["b"] * 3, "quantile": [0.25, 0.5, 0.75], "value": [1.9, 2.0, 2.1]})

    mock_backend.forecast.side_effect = forecast_side_effect

    client.forecast(y_df, None, features=pd.DataFrame(), snapshot_ts=pd.to_datetime("2025-01-01"), run_id="test")

    assert mock_backend.forecast.call_count == 4


def test_budget_aborts_forecast(mock_backend, mock_cache, logger):
    """Test that budget aborts forecast when exceeded."""
    config = TimeGPTConfig(api_mode="online")
    budget = APIBudget(per_run=0)
    client = TimeGPTClient(backend=mock_backend, cache=mock_cache, config=config, logger=logger, budget=budget)

    y_df = pd.DataFrame({"unique_id": ["a"], "ds": [pd.to_datetime("2025-01-01")], "y": [1]})
    mock_cache.get.return_value = None

    result = client.forecast(y_df, None, features=pd.DataFrame(), snapshot_ts=pd.to_datetime("2025-01-01"), run_id="test")

    mock_backend.forecast.assert_not_called()
    assert result.empty


def test_forecast_with_exogenous_features(mock_backend, mock_cache, logger):
    """Test that exogenous features are correctly handled."""
    exog_config = ForecastExogConfig(
        use_exogs=True,
        strict_exog=True,
        hist_exog_list_raw=["hist_exog_1"],
        futr_exog_list_raw=["futr_exog_1"],
    )
    config = TimeGPTConfig(api_mode="online", exog=exog_config)
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
        "timestamp": [pd.to_datetime("2025-01-01", utc=True), pd.to_datetime("2025-01-02", utc=True)],
        "symbol": ["a", "a"],
        "hist_exog_1": [10, 11],
        "futr_exog_1": [100, 101],
        "ignored_col": [99, 98],
    })
    mock_cache.get.return_value = None
    mock_backend.forecast.return_value = pd.DataFrame(
        {"unique_id": ["a"] * 3, "quantile": [0.25, 0.5, 0.75], "value": [0.9, 1.0, 1.1]}
    )

    client.forecast(y_df, x_df, features=features, snapshot_ts=pd.to_datetime("2025-01-01", utc=True), run_id="test")

    mock_backend.forecast.assert_called_once()
    call_args, call_kwargs = mock_backend.forecast.call_args
    y_df_passed = call_args[0]
    x_df_passed = call_args[1]
    hist_exog_list_passed = call_kwargs["hist_exog_list"]

    assert "hist_exog_1" in y_df_passed.columns
    assert "ignored_col" not in y_df_passed.columns
    assert "futr_exog_1" in x_df_passed.columns
    assert hist_exog_list_passed == ["hist_exog_1"]
