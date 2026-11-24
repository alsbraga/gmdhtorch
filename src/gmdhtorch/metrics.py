"""Evaluation metrics built on NumPy and scikit-learn."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import root_mean_squared_error


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute range-normalized Root Mean Squared Error ``RMSE / ptp(y_true)``.

    Returns ``float('nan')`` if inputs contain NaN or Inf.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        return float("nan")
    span = float(np.ptp(yt))
    rmse_val = float(root_mean_squared_error(yt, yp))
    if span == 0.0:
        return 0.0 if rmse_val == 0.0 else float("inf")
    return rmse_val / span
