"""
DMDc loss-curve plot (Equation 2).

Loads the parameter trajectory captured during training, fits DMDc on the
first FIT_FRAC of it, forecasts the rest, then evaluates the configurable
loss L(.; D) on both the real x_k and the DMDc-forecast x_hat_k. Writes a
two-curve comparison to config.LOSS_PLOT.
"""

import numpy as np
import torch

import config as C
from analysis import fit_and_forecast, load_snapshots
from data import get_loaders
from figures import comparison_plot
from log import log
from metrics import LOSSES, loss_at
from model import MLP


def main():
    # 1. Load the parameter trajectory.
    X, U, steps = load_snapshots(C.SNAP_PATH)
    n, m = X.shape
    log("info", f"loaded {C.SNAP_PATH}  X={X.shape}  U={U.shape}  m={m} snapshots")

    # 2. Fit DMDc on the first FIT_FRAC of the trajectory and forecast forward.
    X_pred, split, mdl = fit_and_forecast(X, U, C.FIT_FRAC, C.RANK)
    log("ok",
        f"DMDc fit on first {split}/{m} snapshots, rank={mdl['rank']}  "
        f"(A {mdl['A'].shape}, B {mdl['B'].shape})")

    # 3. Equation 2: turn each parameter vector into a scalar loss on D.
    _, test_loader = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)
    net = MLP(C.ARCH).to(C.DEVICE)
    loss_fn = LOSSES[C.LOSS]

    log("info", f"computing actual loss curve  L(x_k; D)         loss={C.LOSS}")
    L_actual = np.array([loss_at(net, X[:, k],     test_loader, C.DEVICE, loss_fn) for k in range(m)])
    log("info", f"computing DMDc loss curve    L(x_hat_k; D)     loss={C.LOSS}")
    L_pred   = np.array([loss_at(net, X_pred[:, k], test_loader, C.DEVICE, loss_fn) for k in range(m)])

    # 4. Render.
    comparison_plot(
        steps, L_actual, L_pred, split,
        out_path=C.LOSS_PLOT,
        title=f"DMDc forecast of MLP parameters on {C.DATASET.upper()}   "
              f"·   fit on first {split}/{m} snapshots   ·   rank {mdl['rank']}",
        ylabel=f"test {C.LOSS}",
        actual_label=r"actual  $\mathcal{L}(x_k;\mathcal{D})$",
        pred_label=r"DMDc   $\mathcal{L}(\hat{x}_k;\mathcal{D})$",
        summary_label="mean |Δloss|   in-sample = {in_:.4f}    out-of-sample = {out:.4f}",
    )
    log("ok", f"saved figure to {C.LOSS_PLOT}")


if __name__ == "__main__":
    main()
