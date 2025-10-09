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
