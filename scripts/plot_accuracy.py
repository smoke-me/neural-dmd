"""
DMDc parameter-prediction-accuracy plot.

Implements e(k) = ||x_hat_k - x_k|| / ||x_k|| and plots accuracy
= 100 * (1 - e(k)) against training step. Inside the fit region the
curve sits at ~100% (full-rank reconstruction is exact); in the
forecast region it drops as small one-step errors compound.

Reads:  snapshots.npz  + analysis.npz
Writes: dmdc_accuracy.png
"""

import sys
import time
from pathlib import Path

# Make the neural_dmd package importable when this script is launched
# directly (e.g. `python scripts/plot_accuracy.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from neural_dmd import config as C
from neural_dmd.figures import comparison_plot
from neural_dmd.log import banner, log
from neural_dmd.snapshots import Recorder, eval_indices


def main():
    t = time.time()
    banner("plot_accuracy.py start", dataset=C.DATASET)

    snap  = Recorder.load(C.SNAP_PATH)
    cache = np.load(C.ANALYSIS_PATH)
    X_pred    = cache["X_pred"]
    fit_split = int(cache["fit_split"])
    rank      = int(cache["rank"])
    log("info",
        f"loaded snapshots ({snap['X'].shape}) + analysis cache  "
        f"rank={rank}  fit_split={fit_split}")

    # Subsample to the per-forecast_every grid.
    idx = eval_indices(snap["steps"], snap["forecast_every"])
    grid_steps = snap["steps"][idx]
    plot_split = int(np.searchsorted(grid_steps, snap["fit_steps"]))
    log("info",
        f"evaluation grid: {len(idx)} points (every {snap['forecast_every']} steps)  "
        f"plot_split={plot_split}")

    Xg   = snap["X"][:, idx]
    Xp_g = X_pred[:, idx]
    e    = np.linalg.norm(Xp_g - Xg, axis=0) / (np.linalg.norm(Xg, axis=0) + 1e-12)
    acc_pred   = 100.0 * (1.0 - e)
    acc_actual = np.full(len(idx), 100.0)

    banner("render figure", out=C.ACC_PLOT)
    comparison_plot(
        grid_steps, acc_actual, acc_pred, plot_split,
        out_path=C.ACC_PLOT,
        title=f"DMDc parameter-prediction accuracy on {C.DATASET.upper()}   "
              f"·   fit on {fit_split} dense snapshots   "
              f"·   eval on {len(idx)} grid points   ·   rank {rank}",
        ylabel="prediction accuracy  [%]",
        actual_label=r"actual  $x_k$  (reference, 100%)",
        pred_label=r"DMDc   $100\cdot(1 - \|\hat{x}_k - x_k\|/\|x_k\|)$",
        summary_label="mean |Δaccuracy|   in-sample = {in_:.4f}%    out-of-sample = {out:.4f}%",
    )
    log("ok", f"saved {C.ACC_PLOT}  total_elapsed={time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
