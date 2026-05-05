"""
DMDc fit + forecast (Proctor-Brunton-Kutz, SIAM J. Appl. Dyn. Syst. 2016,
augmented with full-X output basis so in-sample reconstruction is exact
at full rank).

Inputs are cast to f64 internally so the SVD is numerically faithful;
the returned X_pred is cast back to f32 to match the snapshot NPZ schema.

Public surface:
    run(snap: dict, *, rank: int | None) -> dict
        Returns {'X_pred': (n, m) f32, 'A': (p, p) f64, 'rank': int,
                 'fit_split': int}.

Module-private (also imported by optdmdc.py / coptdmdc.py):
    _kernel(X_fit, U_fit, *, rank, n, q, m_fit) -> dict
        Returns {
            'Ux'      : (n, p) truncated POD basis used for F, G,
            'Ux_full' : (n, k) un-truncated POD basis (k = min(n, m_fit)),
            'Sx'      : (k,)   full singular value spectrum of X_fit,
            'F'       : (p, p) reduced operator (= reduced A),
            'G'       : (p, q) reduced control matrix (= reduced B),
            'p'       : truncation rank actually used,
            'r_omega' : truncation rank used for Omega = [X1; U_fit] SVD,
            'avail_x' : full available rank of X_fit,
            'avail_o' : full available rank of Omega,
        }
        Reused as the warm start for the variable-projection variants.
"""

import time

import numpy as np

from .log import log, log_spectrum, progress


_SVD_CACHE: dict = {}


def _svd_fingerprint(X_fit, U_fit) -> tuple:
    """Cheap content fingerprint for cache keying.

    Hashes shape + dtype + a few sample values per array. False
    positives are essentially impossible for our snapshot files (which
    are large, dense, content-deterministic), and the fingerprint
    computation is microseconds vs the ~100s SVD it gates.
    """
    def _fp(a):
        flat = a.ravel()
        n = flat.size
        if n == 0:
            return (a.shape, str(a.dtype), 0)
        i_mid = n // 2
        return (a.shape, str(a.dtype),
                float(flat[0]), float(flat[i_mid]), float(flat[-1]))
    return (_fp(X_fit), _fp(U_fit))


def _compute_svds(X_fit, U_fit, *, n, q, m_fit) -> dict:
    """Compute the two SVDs that drive every method's warm start.
    Returns the FULL (untruncated) factors so callers at any rank can
    truncate without re-running an SVD."""
    X1 = X_fit[:, :-1]
    X2 = X_fit[:, 1:]

    log("info", f"_kernel: thin SVD(Omega)  shape=({n + q},{m_fit - 1})")
    t = time.time()
    Omega = np.vstack([X1, U_fit])
    Uo_full, So_full, Vto_full = np.linalg.svd(Omega, full_matrices=False)
    log("ok",
        f"_kernel: SVD(Omega) done in {time.time() - t:.2f}s  "
        f"avail_rank={len(So_full)}")
    log_spectrum("_kernel: spectrum(Omega) [full]", So_full)

    log("info", f"_kernel: thin SVD(X_fit)  shape=({n},{m_fit})")
    t = time.time()
    Ux_full, Sx, _ = np.linalg.svd(X_fit, full_matrices=False)
    log("ok",
        f"_kernel: SVD(X_fit) done in {time.time() - t:.2f}s  "
        f"avail_rank={len(Sx)}")
    log_spectrum("_kernel: spectrum(X_fit) [full]", Sx)

    return {
        "X2":       X2,
        "Uo_full":  Uo_full,
        "So_full":  So_full,
        "Vto_full": Vto_full,
        "Ux_full":  Ux_full,
        "Sx":       Sx,
        "n":        n,
        "q":        q,
        "m_fit":    m_fit,
    }


def _kernel(X_fit, U_fit, *, rank, n, q, m_fit):
    # SVD of input subspace Omega = [X1; U_fit] -> Uo, So, Vto.
    # SVD of X_fit                              -> Ux_full, Sx.
    # Both SVDs are cached by content fingerprint of (X_fit, U_fit) so
    # back-to-back calls within the same run_exp pipeline (DMDc, sDMDc,
    # OptDMDc, cOptDMDc) reuse a single pair of SVDs even when their
    # ranks differ - rank truncation is just slicing afterwards.
    fp = _svd_fingerprint(X_fit, U_fit)
    cached = _SVD_CACHE.get(fp)
    if cached is None:
        log("info", "_kernel: cache MISS, computing SVDs")
        cached = _compute_svds(X_fit, U_fit, n=n, q=q, m_fit=m_fit)
        _SVD_CACHE[fp] = cached
    else:
        log("ok",
            f"_kernel: cache HIT (reusing prior SVDs for X_fit{X_fit.shape}, "
            f"U_fit{U_fit.shape})")

    X2       = cached["X2"]
    Uo_full  = cached["Uo_full"]
    So_full  = cached["So_full"]
    Vto_full = cached["Vto_full"]
    Ux_full  = cached["Ux_full"]
    Sx       = cached["Sx"]

    avail_o = len(So_full)
    r = avail_o if rank is None else min(rank, avail_o)
    avail_x = len(Sx)
    p = avail_x if rank is None else min(rank, avail_x)

    Uo, So, Vto = Uo_full[:, :r], So_full[:r], Vto_full[:r, :]
    Ux = Ux_full[:, :p]

    log("info",
        f"_kernel: assembling reduced operators F (p x p), G (p x q)  "
        f"  used_rank_omega={r}  used_rank_x={p}")
    t = time.time()
    Uo1 = Uo[:n, :]
    Uo2 = Uo[n:, :]
    common = X2 @ Vto.T @ np.diag(1.0 / So)
    F = Ux.T @ common @ Uo1.T @ Ux
    G = Ux.T @ common @ Uo2.T
    log("ok",
        f"_kernel: reduced operators assembled in {time.time() - t:.2f}s  "
        f"F={F.shape}  G={G.shape}")

    return {
        "Ux":      Ux,
        "Ux_full": Ux_full,
        "Sx":      Sx,
        "F":       F,
        "G":       G,
        "p":       p,
        "r_omega": r,
        "avail_x": avail_x,
        "avail_o": avail_o,
    }


def _kernel_cache_clear() -> None:
    """Drop cached SVDs (frees up to several GB depending on m_fit).
    Call between unrelated experiments if running long sessions."""
    _SVD_CACHE.clear()


def run(snap: dict, *, rank: int | None = None) -> dict:
    # We cast the snapshot data to f64 internally so the SVD is
    # numerically faithful. The output X_pred is cast back to f32 at the
    # end so the npz schema downstream stays unchanged.
    X = snap["X"].astype(np.float64, copy=False)              # (n, m)
    U = snap["U"].astype(np.float64, copy=False)              # (q, total_steps - 1)
    steps = snap["steps"]                                     # (m,)
    fit_split     = int(snap["fit_split"])                    # first snap idx OUTSIDE fit window
    fit_start_idx = int(snap.get("fit_start_idx", 0))         # first snap idx INSIDE fit window
    fit_steps     = int(snap["fit_steps"])                    # gradient step where fit window ends
    total_steps   = int(snap["total_steps"])

    n, m = X.shape
    q = U.shape[0]

    # ----- 1. assemble fit data -----
    # Inside the fit window snapshots are taken every fit_every step (= 1 by
    # default), so the gradient-step indices are contiguous and the control
    # that drove x_k -> x_{k+1} is simply U[:, steps[k]].
    X_fit = X[:, fit_start_idx:fit_split]                     # (n, m_fit)
    U_fit = U[:, steps[fit_start_idx:fit_split - 1]]          # (q, m_fit - 1)

    log("info",
        f"DMDc/numpy: stage 1/3  fit data assembled  "
        f"X_fit={X_fit.shape}  U_fit={U_fit.shape}  n={n}  q={q}  "
        f"requested_rank={rank}")
    t0 = time.time()

    # ----- 2. reduced operators (Ux, F, G) via shared kernel -----
    log("info", "DMDc/numpy: stage 2/3  reduced operators via _kernel")
    K = _kernel(X_fit, U_fit, rank=rank, n=n, q=q, m_fit=fit_split)
    Ux = K["Ux"]; F = K["F"]; G = K["G"]; p = K["p"]
    log("ok",
        f"DMDc/numpy: stage 2/3  done in {time.time() - t0:.2f}s  "
        f"rank={p}  A=F={F.shape}  B=G={G.shape}")

    # ----- 3. in-sample reconstruction (project + lift) -----
    log("info",
        f"DMDc/numpy: stage 3/3  in-sample POD lift  "
        f"X[:, {fit_start_idx}:{fit_split}] -> Ux Ux^T X_fit  (rank={p})")
    X_pred = np.empty_like(X)
    X_pred[:, fit_start_idx:fit_split] = Ux @ (Ux.T @ X_fit)
    if fit_start_idx > 0:
        # pre-fit region: project sparse pre-fit snapshots onto the same
        # POD basis Ux so the comparison plot shows POD-truncation
        # behaviour over the whole trajectory, not just the fit window.
        log("info",
            f"DMDc/numpy: pre-fit POD lift  X[:, :{fit_start_idx}] -> Ux Ux^T X[:, :{fit_start_idx}]")
        X_pred[:, :fit_start_idx] = Ux @ (Ux.T @ X[:, :fit_start_idx])

    # ----- 4. forecast: roll forward in reduced space, lift at sample steps -----
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
        y = F @ y + G @ U[:, cur_step]
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
    # F is returned in f64 so the eigenvalue analysis (plot_eigenvalues.py)
    # works at full precision; it's small (rank x rank) so this is cheap.
    return {"X_pred": X_pred.astype(np.float32, copy=False),
            "A":      F,
            "rank":   p,
            "fit_split": fit_split}
