"""
DMDc operator-eigenvalue plot in the complex plane.

The DMDc fit produces a matrix A such that x_{k+1} ~ A x_k + B u_k. The
eigenvalues of A are the modes of the linear dynamics:
  - |lambda| < 1 : that mode decays as we iterate forward
  - |lambda| = 1 : that mode oscillates without growing
  - |lambda| > 1 : that mode grows (forecast unstable along it)
The unit circle (radius 1, dashed) is the stability boundary.

Reads:  analysis.npz   (just the eigenvalues + rank)
Writes: dmdc_eigenvalues.png
"""

import sys
import time
from pathlib import Path

# Make the neural_dmd package importable when this script is launched
# directly (e.g. `python scripts/plot_eigenvalues.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from neural_dmd import config as C
from neural_dmd.figures import eigenvalue_plot
from neural_dmd.log import banner, log


def main():
    t = time.time()
    banner("plot_eigenvalues.py start", dataset=C.DATASET, output=C.EIG_PLOT)

    cache = np.load(C.ANALYSIS_PATH)
    eigvals = cache["eigenvalues"]
    rank    = int(cache["rank"])
    mags    = np.abs(eigvals)
    spectral_radius = float(mags.max()) if mags.size else 0.0

    log("info", f"loaded {len(eigvals)} eigenvalues from {C.ANALYSIS_PATH}  rank={rank}")
    log("info", f"  inside  unit circle (|λ|≤1): {int((mags <= 1.0).sum())}/{len(eigvals)}")
    log("info", f"  outside unit circle (|λ|>1): {int((mags >  1.0).sum())}/{len(eigvals)}")
    log("info", f"  spectral radius max|λ| = {spectral_radius:.6f}")

    eigenvalue_plot(
        eigvals,
        out_path=C.EIG_PLOT,
        title=f"DMDc A-matrix eigenvalues on {C.DATASET.upper()}   "
              f"·   rank {rank}   ·   spectral radius = {spectral_radius:.4f}",
    )
    log("ok", f"saved {C.EIG_PLOT}  total_elapsed={time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
