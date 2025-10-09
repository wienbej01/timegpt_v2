from __future__ import annotations

import numpy as np
import pytest

from timegpt_v2.eval.metrics_forecast import pinball_loss, pit_coverage


@pytest.fixture
def y_true() -> np.ndarray:
    """Return a sample y_true array for testing."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def y_pred() -> np.ndarray:
    """Return a sample y_pred array for testing."""
    return np.array([1.1, 2.2, 2.8, 4.3, 5.1])


@pytest.fixture
def q25() -> np.ndarray:
    """Return a sample q25 array for testing."""
    return np.array([0.9, 1.8, 2.9, 3.8, 4.9])


@pytest.fixture
def q75() -> np.ndarray:
    """Return a sample q75 array for testing."""
    return np.array([1.1, 2.2, 3.1, 4.2, 5.1])


def test_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Test the pinball_loss function."""
    loss = pinball_loss(y_true, y_pred, quantile=0.5)
    assert isinstance(loss, float)
    assert loss > 0


def test_pinball_loss_decreases_with_better_forecast(y_true: np.ndarray) -> None:
    """Test that the pinball loss decreases when the forecast is closer to the truth."""
    y_pred_good = y_true + 0.1
    y_pred_bad = y_true + 1.0
    loss_good = pinball_loss(y_true, y_pred_good, quantile=0.5)
    loss_bad = pinball_loss(y_true, y_pred_bad, quantile=0.5)
    assert loss_good < loss_bad


def test_pit_coverage(y_true: np.ndarray, q25: np.ndarray, q75: np.ndarray) -> None:
    """Test the pit_coverage function."""
    coverage = pit_coverage(y_true, q25, q75)
    assert isinstance(coverage, float)
    assert 0.0 <= coverage <= 1.0


def test_pit_coverage_on_synthetic_data() -> None:
    """Test the pit_coverage function on synthetic data."""
    y_true = np.random.normal(size=1000)
    q25 = y_true - 0.5
    q75 = y_true + 0.5
    coverage = pit_coverage(y_true, q25, q75)
    assert coverage == 1.0

    q25 = y_true + 0.1
    q75 = y_true + 0.5
    coverage = pit_coverage(y_true, q25, q75)
    assert coverage == 0.0
