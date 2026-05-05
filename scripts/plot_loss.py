"""
DMDc loss-curve comparison plot.

Reads:  snapshots.npz  + analysis.npz
Writes: dmdc_loss.png  - actual loss vs DMDc-predicted loss at the
                        per-forecast_every evaluation grid.
"""

import sys
import time
from pathlib import Path

# Make the neural_dmd package importable when this script is launched
# directly (e.g. `python scripts/plot_loss.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch  # noqa: F401  (imported for CUDA detection side-effects via config)

from neural_dmd import config as C
from neural_dmd.data import get_loaders
from neural_dmd.figures import comparison_plot
from neural_dmd.log import banner, log, progress
from neural_dmd.metrics import LOSSES, loss_at
from neural_dmd.model import MLP
from neural_dmd.snapshots import Recorder, eval_indices


def _eval_curve(label, net, X_cols, idx, test_loader, loss_fn):
    # Walk one parameter trajectory and evaluate the loss at each
    # subsampled column. Logs progress every 25%.
    t = time.time()
    out = np.empty(len(idx), dtype=np.float64)
    for i, k in enumerate(idx, start=1):
        out[i - 1] = loss_at(net, X_cols[:, k], test_loader, C.DEVICE, loss_fn)
        progress(f"{label} loss curve", i, len(idx), t, every_pct=25.0)
    return out


def main():
    t_total = time.time()
    banner("plot_loss.py start", dataset=C.DATASET, loss=C.LOSS, device=C.DEVICE)

    snap  = Recorder.load(C.SNAP_PATH)
    cache = np.load(C.ANALYSIS_PATH)
    X_pred    = cache["X_pred"]
    fit_split = int(cache["fit_split"])
    rank      = int(cache["rank"])
    log("info",
        f"loaded snapshots ({snap['X'].shape}) + analysis cache  "
        f"rank={rank}  fit_split={fit_split}")

    # Subsample to a uniform per-forecast_every grid for evaluation.
    idx = eval_indices(snap["steps"], snap["forecast_every"])
    grid_steps = snap["steps"][idx]
    plot_split = int(np.searchsorted(grid_steps, snap["fit_steps"]))
    log("info",
        f"evaluation grid: {len(idx)} points (every {snap['forecast_every']} steps)  "
        f"plot_split={plot_split}  in-fit={plot_split}  in-forecast={len(idx) - plot_split}")

    # Compute loss curves.
    banner("evaluate loss curves", grid_points=len(idx),
           dataset=C.DATASET, loss=C.LOSS)
    _, test_loader = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)
    net = MLP(C.ARCH).to(C.DEVICE)
    loss_fn = LOSSES[C.LOSS]

    L_actual = _eval_curve("actual", net, snap["X"], idx, test_loader, loss_fn)
    L_pred   = _eval_curve("DMDc",   net, X_pred,    idx, test_loader, loss_fn)

    # Render.
    banner("render figure", out=C.LOSS_PLOT)
    comparison_plot(
        grid_steps, L_actual, L_pred, plot_split,
        out_path=C.LOSS_PLOT,
        title=f"DMDc loss-curve forecast on {C.DATASET.upper()}   "
              f"·   fit on {fit_split} dense snapshots   "
              f"·   eval on {len(idx)} grid points   ·   rank {rank}",
        ylabel=f"test {C.LOSS}",
        actual_label=r"actual  $\mathcal{L}(x_k;\mathcal{D})$",
        pred_label=r"DMDc   $\mathcal{L}(\hat{x}_k;\mathcal{D})$",
        summary_label="mean |Δloss|   in-sample = {in_:.4f}    out-of-sample = {out:.4f}",
    )
    log("ok", f"saved {C.LOSS_PLOT}  total_elapsed={time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
