"""
Run the DMDc fit + forecast once and cache the results so the three
plot_*.py scripts don't each have to redo the expensive SVD.

Reads:  snapshots.npz  (written by train.py)
Writes: analysis.npz   with
            X_pred       (n, m) f32  - predicted parameter trajectory
            eigenvalues  (rank,)  c128 - eigenvalues of the DMDc operator A
            rank         scalar i64
            fit_split    scalar i64
"""

import sys
import time
from pathlib import Path

# Make the neural_dmd package importable when this script is launched
# directly (e.g. `python scripts/analyze.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from neural_dmd import config as C
from neural_dmd.dmdc import run as dmdc_run
from neural_dmd.log import banner, log
from neural_dmd.snapshots import Recorder


def main():
    t = time.time()
    banner("analyze.py start", dataset=C.DATASET, snapshots=C.SNAP_PATH,
           output=C.ANALYSIS_PATH)

    snap = Recorder.load(C.SNAP_PATH)
    log("info",
        f"loaded {C.SNAP_PATH}  X={snap['X'].shape}  fit_split={snap['fit_split']}  "
        f"total_steps={snap['total_steps']}")

    out = dmdc_run(snap, rank=C.RANK)

    log("info", f"computing eigenvalues of A {out['A'].shape}")
    eigvals = np.linalg.eigvals(out["A"]).astype(np.complex128)
    mags = np.abs(eigvals)
    log("ok",
        f"eigenvalues: {len(eigvals)} total  "
        f"inside_unit_circle={int((mags <= 1.0).sum())}  "
        f"outside={int((mags > 1.0).sum())}  "
        f"spectral_radius={mags.max():.6f}")

    np.savez(
        C.ANALYSIS_PATH,
        X_pred=out["X_pred"],
        eigenvalues=eigvals,
        rank=np.int64(out["rank"]),
        fit_split=np.int64(out["fit_split"]),
    )
    log("ok",
        f"saved {C.ANALYSIS_PATH}  "
        f"(X_pred {out['X_pred'].shape}, eigenvalues {eigvals.shape})  "
        f"total_elapsed={time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
