"""
Evaluation script. Two outputs:

  1. Trained-model report: the configurable METRIC (default: classification
     accuracy) of the saved checkpoint on the held-out test set.

  2. DMDc parameter-prediction-accuracy plot. Implements
                e(k) = ||x_hat_k - x_k||_2 / ||x_k||_2
     and plots accuracy = 100 * (1 - e(k)) against training step. The
     "actual" reference is trivially 100% (a vector compared against
     itself). The "DMDc" curve sits at 100% inside the fit region (where
     x_hat_k matches x_k by construction) and drops in the forecast region
     as the linear model extrapolates. Saved to config.ACC_PLOT.
"""

import numpy as np
import torch

import config as C
from analysis import fit_and_forecast, load_snapshots
from data import get_loaders
from figures import comparison_plot
from log import log
from metrics import METRICS, metric_at
from model import MLP
from params import flatten_params


def main():
    _, test_loader = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)
    metric_fn = METRICS[C.METRIC]

    # --- 1. Trained-model evaluation on the held-out test set. ---
    log("info", f"loading {C.CKPT_PATH} on {C.DEVICE}")
    net = MLP(C.ARCH).to(C.DEVICE)
    net.load_state_dict(torch.load(C.CKPT_PATH, map_location=C.DEVICE))

    score = metric_at(net, flatten_params(net), test_loader, C.DEVICE, metric_fn)
    log("ok", f"trained-model {C.METRIC} = {score*100:.2f}%")

    # --- 2. DMDc parameter-prediction-accuracy plot. ---
    X, U, steps = load_snapshots(C.SNAP_PATH)
    m = X.shape[1]
    log("info", f"loaded {C.SNAP_PATH}  X={X.shape}  U={U.shape}  m={m} snapshots")

    X_pred, split, mdl = fit_and_forecast(X, U, C.FIT_FRAC, C.RANK)
    log("ok", f"DMDc fit on first {split}/{m} snapshots, rank={mdl['rank']}")

    # e(k) = ||x_hat_k - x_k|| / ||x_k||  -> prediction accuracy = 100*(1 - e).
    e = np.linalg.norm(X_pred - X, axis=0) / (np.linalg.norm(X, axis=0) + 1e-12)
    acc_pred   = 100.0 * (1.0 - e)
    acc_actual = np.full(m, 100.0)               # a vector vs itself: 0 error

    comparison_plot(
        steps, acc_actual, acc_pred, split,
        out_path=C.ACC_PLOT,
        title=f"DMDc parameter prediction accuracy on {C.DATASET.upper()}   "
              f"·   fit on first {split}/{m} snapshots   ·   rank {mdl['rank']}",
        ylabel="prediction accuracy  [%]",
        actual_label=r"actual  $x_k$  (reference, 100%)",
        pred_label=r"DMDc   $100\cdot(1 - \|\hat{x}_k - x_k\|/\|x_k\|)$",
        summary_label="mean |Δaccuracy|   in-sample = {in_:.4f}%    out-of-sample = {out:.4f}%",
    )
    log("ok", f"saved figure to {C.ACC_PLOT}")


if __name__ == "__main__":
    main()
