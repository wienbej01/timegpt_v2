import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

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

    client.forecast(y_df, None, snapshot_ts=pd.to_datetime("2025-01-01"))

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

    client.forecast(y_df, None, snapshot_ts=pd.to_datetime("2025-01-01"))

    mock_backend.forecast.assert_called_once()
    mock_cache.put.assert_called_once()


def test_retry_on_payload_too_large(mock_backend, mock_cache, logger):
    """Test retry logic on PayloadTooLargeError."""
    config = TimeGPTConfig(api_mode="online")
    client = TimeGPTClient(backend=mock_backend, cache=mock_cache, config=config, logger=logger)

    y_df = pd.DataFrame({"unique_id": ["a", "b"], "ds": [pd.to_datetime("2025-01-01")] * 2, "y": [1, 2]})
    mock_cache.get.return_value = None
    mock_backend.forecast.side_effect = [
        PayloadTooLargeError,
        pd.DataFrame({"unique_id": ["a"] * 3, "quantile": [0.25, 0.5, 0.75], "value": [0.9, 1.0, 1.1]}),
        pd.DataFrame({"unique_id": ["b"] * 3, "quantile": [0.25, 0.5, 0.75], "value": [1.9, 2.0, 2.1]}),
    ]

    client.forecast(y_df, None, snapshot_ts=pd.to_datetime("2025-01-01"))

    assert mock_backend.forecast.call_count == 3


def test_budget_aborts_forecast(mock_backend, mock_cache, logger):
    """Test that budget aborts forecast when exceeded."""
    config = TimeGPTConfig(api_mode="online")
    budget = APIBudget(per_run=0)
    client = TimeGPTClient(backend=mock_backend, cache=mock_cache, config=config, logger=logger, budget=budget)

    y_df = pd.DataFrame({"unique_id": ["a"], "ds": [pd.to_datetime("2025-01-01")], "y": [1]})
    mock_cache.get.return_value = None

    result = client.forecast(y_df, None, snapshot_ts=pd.to_datetime("2025-01-01"))

    mock_backend.forecast.assert_not_called()
    assert result.empty
