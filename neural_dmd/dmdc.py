"""
DMDc fit + forecast (Proctor-Brunton-Kutz, SIAM J. Appl. Dyn. Syst. 2016,
augmented with full-X output basis so in-sample reconstruction is exact
at full rank).

Inputs are cast to f64 internally so the SVD is numerically faithful;
the returned X_pred is cast back to f32 to match the snapshot NPZ schema.

Public surface:
    run(snap: dict, *, rank: int | None) -> dict
        Returns {'X_pred': (n, m) f32, 'rank': int, 'fit_split': int}.
"""

import time

import numpy as np

from .log import log, progress


def run(snap: dict, *, rank: int | None = None) -> dict:
    # We cast the snapshot data to f64 internally so the SVD is
    # numerically faithful. The output X_pred is cast back to f32 at the
    # end so the npz schema downstream stays unchanged. Memory cost:
    # X_fit doubles from ~657MB to ~1.3GB; SVD slows by ~2x
    # (dgesdd vs sgesdd).
    X = snap["X"].astype(np.float64, copy=False)              # (n, m)
    U = snap["U"].astype(np.float64, copy=False)              # (q, total_steps - 1)
    steps = snap["steps"]                                     # (m,)
    fit_split = int(snap["fit_split"])
    fit_steps = int(snap["fit_steps"])
    total_steps = int(snap["total_steps"])

    n, m = X.shape
    q = U.shape[0]

    # ----- 1. assemble fit data -----
    # Inside the fit region snapshots are taken every fit_every step (= 1 by
    # default), so the gradient-step indices are contiguous and the control
    # that drove x_k -> x_{k+1} is simply U[:, steps[k]].
    X_fit = X[:, :fit_split]                                  # (n, fit_split)
    U_fit = U[:, steps[:fit_split - 1]]                       # (q, fit_split - 1)

    log("info",
        f"DMDc/numpy: stage 1/4  fitting on {fit_split} snapshots  n={n}  q={q}  "
        f"requested_rank={rank}")
    t0 = time.time()

    X1 = X_fit[:, :-1]                                        # (n, fit_split - 1)
    X2 = X_fit[:, 1:]                                         # (n, fit_split - 1)

    # ----- 2. SVD of input subspace Omega = [X1; U_fit] -----
    log("info", f"DMDc/numpy: stage 2/4  thin SVD(Omega) shape ({n + q},{fit_split - 1})")
    t_svd_o = time.time()
    Omega = np.vstack([X1, U_fit])                            # (n + q, fit_split - 1)
    Uo, So, Vto = np.linalg.svd(Omega, full_matrices=False)
    r = len(So) if rank is None else min(rank, len(So))
    Uo, So, Vto = Uo[:, :r], So[:r], Vto[:r, :]
    log("ok",
        f"DMDc/numpy: stage 2/4  SVD(Omega) done in {time.time() - t_svd_o:.2f}s  "
        f"avail_rank={len(So)}  used_rank={r}")

    # ----- 3. SVD of output basis (use full X_fit so x_0 lies in span(Ux)) -----
    log("info", f"DMDc/numpy: stage 3/4  thin SVD(X_fit) shape ({n},{fit_split})")
    t_svd_x = time.time()
    Ux, Sx, _ = np.linalg.svd(X_fit, full_matrices=False)
    p = len(Sx) if rank is None else min(rank, len(Sx))
    Ux = Ux[:, :p]                                            # (n, p)
    log("ok",
        f"DMDc/numpy: stage 3/4  SVD(X_fit) done in {time.time() - t_svd_x:.2f}s  "
        f"avail_rank={len(Sx)}  used_rank={p}")

    # ----- 4. reduced operators -----
    Uo1 = Uo[:n, :]                                           # (n, r)
    Uo2 = Uo[n:, :]                                           # (q, r)
    common = X2 @ Vto.T @ np.diag(1.0 / So)                   # (n, r)
    A = Ux.T @ common @ Uo1.T @ Ux                            # (p, p)
    B = Ux.T @ common @ Uo2.T                                 # (p, q)

    log("ok",
        f"DMDc/numpy: stage 4/4  reduced operators ready in {time.time() - t0:.2f}s  "
        f"rank={p}  A={A.shape}  B={B.shape}")

    # ----- 5. in-sample reconstruction (project + lift) -----
    # X_pred allocated in f64 to match the rest of the pipeline; cast to
    # f32 at the very end so the npz schema downstream is unchanged.
    X_pred = np.empty_like(X)
    X_pred[:, :fit_split] = Ux @ (Ux.T @ X_fit)

    # ----- 6. forecast: roll forward in reduced space, lift at sample steps -----
    n_iters_total = total_steps - fit_steps
    log("info",
        f"DMDc/numpy: forecasting {n_iters_total} reduced-space iterations  "
        f"sampling {len(steps) - fit_split} predictions")
    t1 = time.time()
    y = Ux.T @ X[:, fit_split - 1]                            # reduced state at step fit_steps - 1
    cur_step = fit_steps - 1
    targets = steps[fit_split:]                               # gradient steps to sample
    target_idx = 0
    iters = 0

    while target_idx < len(targets) and cur_step < total_steps - 1:
        y = A @ y + B @ U[:, cur_step]
        cur_step += 1
        iters += 1
        if cur_step == targets[target_idx]:
            X_pred[:, fit_split + target_idx] = Ux @ y
            target_idx += 1
        progress("forecast iter", iters, n_iters_total, t1, every_pct=25.0)

    log("ok",
        f"DMDc/numpy: forecast iterated {iters} steps, sampled "
        f"{target_idx} predictions in {time.time() - t1:.2f}s")

    # Cast back to f32 to match the snapshot/forecast NPZ schema.
    # A is returned in f64 so the eigenvalue analysis (plot_eigenvalues.py)
    # works at full precision; it's small (rank x rank) so this is cheap.
    return {"X_pred": X_pred.astype(np.float32, copy=False),
            "A":      A,
            "rank":   p,
            "fit_split": fit_split}
