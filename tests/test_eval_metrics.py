from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from timegpt_v2.eval.metrics_forecast import (
    interval_width_stats,
    pinball_loss,
    pit_coverage,
)
from timegpt_v2.eval.metrics_trading import (
    hit_rate,
    max_drawdown,
    per_symbol_metrics,
    portfolio_hit_rate,
    portfolio_max_drawdown,
    portfolio_sharpe,
    portfolio_total_pnl,
    sharpe_ratio,
)


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


def test_interval_width_stats(q25: np.ndarray, q75: np.ndarray) -> None:
    """Ensure interval width stats compute mean and median correctly."""
    mean_width, median_width = interval_width_stats(q25, q75)
    expected_widths = q75 - q25
    assert mean_width == pytest.approx(expected_widths.mean())
    assert median_width == pytest.approx(np.median(expected_widths))


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """Return a sample trades DataFrame for testing."""
    return pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT", "MSFT"],
        "entry_ts": pd.to_datetime(["2023-01-01T10:00:00Z", "2023-01-02T10:00:00Z", "2023-01-01T10:00:00Z", "2023-01-02T10:00:00Z"]),
        "exit_ts": pd.to_datetime(["2023-01-01T15:00:00Z", "2023-01-02T15:00:00Z", "2023-01-01T15:00:00Z", "2023-01-02T15:00:00Z"]),
        "net_pnl": [100.0, -50.0, 200.0, 150.0],
        "phase": ["in_sample", "oos", "in_sample", "oos"],
    })


@pytest.fixture
def sample_pnl_series() -> pd.Series:
    """Return a sample PnL series for testing."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    pnl = pd.Series([10, -5, 15, 20, -10, 5, 25, -15, 30, 10], index=dates)
    return pnl


def test_portfolio_sharpe(sample_trades: pd.DataFrame) -> None:
    """Test portfolio Sharpe ratio calculation."""
    sharpe = portfolio_sharpe(sample_trades)
    assert isinstance(sharpe, float)


def test_portfolio_max_drawdown(sample_trades: pd.DataFrame) -> None:
    """Test portfolio max drawdown calculation."""
    dd = portfolio_max_drawdown(sample_trades)
    assert isinstance(dd, float)
    assert dd <= 0  # Drawdown is negative or zero


def test_portfolio_hit_rate(sample_trades: pd.DataFrame) -> None:
    """Test portfolio hit rate calculation."""
    hr = portfolio_hit_rate(sample_trades)
    assert isinstance(hr, float)
    assert 0.0 <= hr <= 1.0


def test_portfolio_total_pnl(sample_trades: pd.DataFrame) -> None:
    """Test portfolio total PnL calculation."""
    pnl = portfolio_total_pnl(sample_trades)
    assert isinstance(pnl, float)
    assert pnl == 400.0  # Sum of all net_pnl


def test_per_symbol_metrics(sample_trades: pd.DataFrame) -> None:
    """Test per-symbol metrics calculation."""
    metrics = per_symbol_metrics(sample_trades)
    assert isinstance(metrics, pd.DataFrame)
    assert not metrics.empty
    expected_columns = ["symbol", "trade_count", "total_net_pnl", "hit_rate", "sharpe", "max_drawdown"]
    assert all(col in metrics.columns for col in expected_columns)


def test_portfolio_metrics_by_phase(sample_trades: pd.DataFrame) -> None:
    """Test portfolio metrics filtered by phase."""
    oos_trades = sample_trades[sample_trades["phase"] == "oos"]
    sharpe_oos = portfolio_sharpe(oos_trades)
    assert isinstance(sharpe_oos, float)
    hr_oos = portfolio_hit_rate(oos_trades)
    assert isinstance(hr_oos, float)
    assert hr_oos == 0.5  # One win, one loss
