"""
Shared helpers for the DMDc post-training analysis pipeline:
    load_snapshots(path)               -> (X, U, steps)
    fit_and_forecast(X, U, frac, rank) -> (X_pred, split, model_dict)

Used by both plot.py (loss comparison) and eval.py (accuracy comparison).
"""

import numpy as np

from dmdc import fit_dmdc, forecast


def load_snapshots(path: str):
    # Returns (X, U, steps) as written by snapshots.Recorder.save.
    npz = np.load(path)
    return npz["X"], npz["U"], npz["steps"]


def fit_and_forecast(X: np.ndarray, U: np.ndarray, fit_frac: float, rank=None):
    # Fit DMDc on the first `fit_frac` of the trajectory, then forecast the
    # full length using the recorded control sequence. Returns:
    #   X_pred : (n, m) predicted trajectory (in-sample reconstruction +
    #            out-of-sample extrapolation)
    #   split  : index marking the boundary between fit and forecast
    #   model  : dict with reduced operators {A, B, modes, rank}
    m = X.shape[1]
    split = max(2, int(m * fit_frac))
    model = fit_dmdc(X[:, :split], U[:, :split - 1], rank=rank)
    X_pred = forecast(model, X[:, 0], U)
    return X_pred, split, model
