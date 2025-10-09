from __future__ import annotations

import numpy as np


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the reliability curve."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])

    bin_sums = np.bincount(bin_ids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(bin_ids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(bin_ids, minlength=len(bins))

    # Avoid division by zero
    nonzero = bin_total > 0
    prob_true = np.full(len(bins), 0.0)
    prob_pred = np.full(len(bins), 0.0)
    prob_true[nonzero] = bin_true[nonzero] / bin_total[nonzero]
    prob_pred[nonzero] = bin_sums[nonzero] / bin_total[nonzero]

    return prob_pred, prob_true
