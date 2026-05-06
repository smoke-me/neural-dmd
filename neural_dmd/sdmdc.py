"""
sDMDc - Stable DMDc (DMDc with post-hoc eigenvalue clipping).

Same warm start as dmdc.run (shared _kernel), then a single radial
projection of the reduced-operator eigenvalues onto the closed unit
disk - paper Algorithm 1, line 5 (Rains et al., JCP 2024) applied
ONCE without any LM optimization:

    F = Z diag(mu) Z^-1                    # warm-start eigendecomp
    gamma = log(mu) / dt                   # continuous-time eigenvalues
    gamma_clipped = min(Re(gamma), 0) + Im(gamma) * i
    mu_clipped    = exp(gamma_clipped * dt)
    F_clipped     = Z diag(mu_clipped) Z^-1

Then forecast with F_clipped exactly like dmdc.run does. No variable
projection, no Levenberg-Marquardt, no Jacobian. The whole method is
DMDc plus three lines of clipping.

Useful when DMDc has a near-unit-circle outlier and the underlying
system is known to be physically stable / converged: clipping prevents
forecast blow-up over long horizons without paying the full
optimisation cost of cOptDMDc.

Public surface (mirrors dmdc.run):
    run(snap, *, rank=None, dt=1.0)
        Returns {'X_pred': (n, m) f32, 'A': (p, p) c128,
                 'eigenvalues': (p,) c128, 'gamma': (p,) c128,
                 'rank': int, 'fit_split': int}.
"""

import time

import numpy as np

from . import config as C
from .coptdmdc import _radial_project        # reuse the same one-line clip
from .dmdc import _kernel
from .log import log, progress


def run(snap: dict, *, rank: int | None = None, dt: float = 1.0) -> dict:
    dtype = np.dtype(C.PRECISION)
    X = snap["X"].astype(dtype, copy=False)
    U = snap["U"].astype(dtype, copy=False)
    steps = snap["steps"]
    fit_split     = int(snap["fit_split"])
    fit_start_idx = int(snap.get("fit_start_idx", 0))
    fit_steps     = int(snap["fit_steps"])
    total_steps   = int(snap["total_steps"])

    n, m = X.shape
    q = U.shape[0]

    X_fit = X[:, fit_start_idx:fit_split]                        # (n, m_fit)
    U_fit = U[:, steps[fit_start_idx:fit_split - 1]]             # (q, m_fit-1)

    log("info",
        f"sDMDc: stage 1/3  fit data assembled  X_fit={X_fit.shape}  "
        f"U_fit={U_fit.shape}  n={n}  q={q}  rank={rank}")
    t0 = time.time()

    # ----- 1. warm-start via shared dmdc kernel -----
    log("info", "sDMDc: stage 1/3  warm-starting via DMDc kernel")
    K = _kernel(X_fit, U_fit, rank=rank, n=n, q=q, m_fit=fit_split)
    Ux = K["Ux"]; F0 = K["F"]; G = K["G"]; p = K["p"]
    log("ok",
        f"sDMDc: stage 1/3  warm start in {time.time() - t0:.2f}s  "
        f"rank={p}  F0={F0.shape}  G={G.shape}")

    # ----- 2. eigendecomposition + radial projection (one shot, no LM) -----
    log("info", "sDMDc: stage 2/3  eigendecomposition + radial projection")
    t = time.time()
    mu, Z = np.linalg.eig(F0)                                    # F0 = Z diag(mu) Z^-1
    Z_inv = np.linalg.inv(Z)
    gamma_raw = np.log(mu).astype(complex) / dt
    n_unstable = int((gamma_raw.real > 0).sum())
    gamma = _radial_project(gamma_raw)                           # min(Re,0) + Im*i
    mu_clipped = np.exp(gamma * dt)
    F_new = Z @ np.diag(mu_clipped) @ Z_inv                      # (p, p) complex
    log("ok",
        f"sDMDc: stage 2/3  eig+clip done in {time.time() - t:.2f}s  "
        f"spectral_radius_raw={np.max(np.abs(mu)):.6f}  "
        f"clipped_unstable={n_unstable}/{p}  "
        f"spectral_radius_new={np.max(np.abs(mu_clipped)):.6f}  "
        f"max_real_gamma={gamma.real.max():.3e}")

    # ----- 3. in-sample reconstruction + forecast -----
    log("info",
        f"sDMDc: stage 3/3  in-sample POD lift  "
        f"X[:, {fit_start_idx}:{fit_split}] -> Ux Ux^T X_fit  (rank={p})  +  forecast")
    X_pred = np.empty_like(X)
    X_pred[:, fit_start_idx:fit_split] = Ux @ (Ux.T @ X_fit)
    if fit_start_idx > 0:
        X_pred[:, :fit_start_idx] = Ux @ (Ux.T @ X[:, :fit_start_idx])

    n_iters_total = total_steps - fit_steps
    log("info",
        f"sDMDc: forecasting {n_iters_total} reduced-space iterations  "
        f"sampling {len(steps) - fit_split} predictions")
    tF = time.time()
    y = (Ux.T @ X[:, fit_split - 1]).astype(complex)
    cur_step = fit_steps - 1
    targets = steps[fit_split:]
    target_idx = 0
    iters = 0
    while target_idx < len(targets) and cur_step < total_steps - 1:
        y = F_new @ y + G @ U[:, cur_step]
        cur_step += 1
        iters += 1
        if cur_step == targets[target_idx]:
            X_pred[:, fit_split + target_idx] = (Ux @ y).real
            target_idx += 1
        progress("forecast iter", iters, n_iters_total, tF, every_pct=25.0)
    log("ok",
        f"sDMDc: forecast iterated {iters} steps, sampled "
        f"{target_idx} predictions in {time.time() - tF:.2f}s")

    return {"X_pred":      X_pred.astype(np.float32, copy=False),
            "A":           F_new,
            "eigenvalues": mu_clipped,
            "gamma":       gamma,
            "rank":        p,
            "fit_split":   fit_split}
