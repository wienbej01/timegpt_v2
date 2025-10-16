from __future__ import annotations

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Calculate the pinball loss."""
    delta = y_true - y_pred
    return np.maximum(quantile * delta, (quantile - 1) * delta).mean()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Error."""
    return np.abs(y_true - y_pred).mean()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def rmae(y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray) -> float:
    """Calculate the Relative Mean Absolute Error."""
    return mae(y_true, y_pred) / mae(y_true, y_persistence)


def rrmse(y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray) -> float:
    """Calculate the Relative Root Mean Squared Error."""
    return rmse(y_true, y_pred) / rmse(y_true, y_persistence)


def pit_coverage(y_true: np.ndarray, q25: np.ndarray, q75: np.ndarray) -> float:
    """Calculate the PIT coverage."""
    return float(np.mean((y_true >= q25) & (y_true <= q75)))


def interval_width_stats(q25: np.ndarray, q75: np.ndarray) -> tuple[float, float]:
    """Return mean and median forecast interval widths."""
    widths = np.asarray(q75, dtype=float) - np.asarray(q25, dtype=float)
    if widths.size == 0:
        return 0.0, 0.0
    return float(widths.mean()), float(np.median(widths))


def crps(y_true: np.ndarray, q25: np.ndarray, q50: np.ndarray, q75: np.ndarray) -> float:
    """Calculate the Continuous Ranked Probability Score using quantile approximation.

    This is a simplified CRPS calculation using three quantiles (25th, 50th, 75th)
    to approximate the full distribution. The formula approximates the integral
    of the squared difference between the CDF of forecasts and the step function
    at the observation.

    Args:
        y_true: True values
        q25: 25th percentile forecasts
        q50: 50th percentile (median) forecasts
        q75: 75th percentile forecasts

    Returns:
        CRPS value (lower is better)
    """
    y_true = np.asarray(y_true)
    q25 = np.asarray(q25)
    q50 = np.asarray(q50)
    q75 = np.asarray(q75)

    if y_true.size == 0:
        return 0.0

    # Simple quantile-based CRPS approximation
    # Weight sum of absolute errors at different quantiles
    crps_value = (
        0.25 * np.abs(y_true - q25) +
        0.5 * np.abs(y_true - q50) +
        0.25 * np.abs(y_true - q75)
    ).mean()

    # Add penalty for quantile crossing (if any)
    crossing_penalty = np.maximum(0, q25 - q50).mean() + np.maximum(0, q50 - q75).mean()
    crps_value += crossing_penalty

    return float(crps_value)


def information_coefficient(forecasts: np.ndarray, actuals: np.ndarray) -> float:
    """Calculate the Information Coefficient (rank correlation).

    The IC measures the rank correlation between forecast values and actual returns.
    It's a key metric for evaluating forecast skill in quantitative finance.

    Args:
        forecasts: Forecast values (typically median forecasts)
        actuals: Actual realized values

    Returns:
        IC value (Spearman rank correlation)
    """
    from scipy.stats import spearmanr

    forecasts = np.asarray(forecasts)
    actuals = np.asarray(actuals)

    if forecasts.size == 0 or actuals.size == 0:
        return 0.0

    if forecasts.size != actuals.size:
        raise ValueError("Forecasts and actuals must have the same size")

    # Remove any NaN values
    valid_mask = ~(np.isnan(forecasts) | np.isnan(actuals))
    if valid_mask.sum() < 2:
        return 0.0

    forecasts_valid = forecasts[valid_mask]
    actuals_valid = actuals[valid_mask]

    # Calculate Spearman rank correlation
    correlation, _ = spearmanr(forecasts_valid, actuals_valid)
    return float(correlation) if not np.isnan(correlation) else 0.0
