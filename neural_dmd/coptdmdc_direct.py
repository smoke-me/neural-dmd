"""
cOptDMDc (direct-λ form) - stability-constrained variant of
neural_dmd.optdmdc_direct.

Constraint
----------
Discrete-time stability is |λ| ≤ 1 for every eigenvalue. The log-form
cOptDMDc (neural_dmd.coptdmdc) exploits the fact that
    |λ| ≤ 1  ⇔  Re(γ) ≤ 0
to recast the constraint as a *linear* inequality on Re(δ), which
scipy.optimize.lsq_linear solves with linear-inequality bounds. In the
direct λ form there is no log; the constraint becomes the curved (non-
linear) "stay inside the unit disk" condition on λ_new = λ - δ.

We don't try to solve the constrained subproblem exactly. Instead:

    1. Solve the unconstrained LM subproblem for δ (same as
       optdmdc_direct).
    2. Form λ_trial = λ - δ.
    3. Radially project each λ_trial[i] back onto the closed unit disk:
           if |λ| > 1: λ ← λ / |λ|       (snap to the unit circle)
       This is the L2-closest point in the disk; matches the spirit of
       the log-form's Re(γ) clip onto 0 (which maps unstable γ onto the
       imaginary axis = the unit circle in λ).
    4. Evaluate residual at the projected λ. The outer LM accept/reject
       + trust-region damping (nu) then handle whether the (post-
       projection) step was actually progress.

Practical consequences
----------------------
- Strict convergence guarantees of constrained VarPro are weaker;
  projection can lengthen the path. In practice the warm-start is
  already close to feasible, and the trust-region damping converges
  rapidly.
- No scipy.optimize.lsq_linear call - the inner solve is the same
  numpy.linalg.lstsq path optdmdc_direct uses. Faster per inner
  attempt.
- Numerically robust: tiny |λ| ≈ 0 are perfectly fine (cause underflow
  to 0 in the forward recurrence, not NaN-via-exp-overflow). This is
  what made cOptDMDc fail under per_tensor + Gavish-Donoho rank=144.

Public surface mirrors neural_dmd.coptdmdc.run.
"""

import time

import numpy as np

from . import config as C
from .dmdc import _kernel
from .log import log
from .optdmdc import (EarlyStopper, _jacobian_columns,
                      _solve_lm_unconstrained)
from .optdmdc_direct import _build_psi_and_dpsi_direct


def _disk_project(lam: np.ndarray) -> np.ndarray:
    """Radial projection onto the closed unit disk in C. For each
    component λ_i: if |λ_i| > 1, replace it with λ_i / |λ_i| (= scale
    to the unit circle, preserving phase). Stable: |λ_i| ≤ 1 entries
    pass through unchanged; |λ_i| = 0 is left at 0."""
    mag = np.abs(lam)
    scale = np.where(mag > 1.0, 1.0 / np.where(mag > 0, mag, 1.0), 1.0)
    return lam * scale


def run(snap: dict, *, rank: int | None = None, pod_rank: int | None = None,
        max_iter: int = 50, tol: float = 1e-6, gmax: int = 50,
        incr: float = 1.5, decr: float = 2.0, nu0: float = 2.0,
        dt: float = 1.0,
        tol_rel: float = 1e-3, patience: int = 3) -> dict:
    del dt   # absorbed into λ; accepted for API symmetry with coptdmdc.run

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

    X_fit = X[:, fit_start_idx:fit_split]
    U_fit = U[:, steps[fit_start_idx:fit_split - 1]]

    log("info",
        f"cOptDMDc(direct): stage 1/4  fit data assembled  X_fit={X_fit.shape}  "
        f"U_fit={U_fit.shape}  n={n}  q={q}  rank={rank}  pod_rank={pod_rank}")
    t0 = time.time()

    # ----- 1. warm-start via shared dmdc kernel -----
    log("info", "cOptDMDc(direct): stage 1/4  warm-starting via DMDc kernel")
    K = _kernel(X_fit, U_fit, rank=rank, n=n, q=q, m_fit=fit_split)
    Ux = K["Ux"]; F0 = K["F"]; G = K["G"]; p = K["p"]
    Ux_full = K["Ux_full"]; avail_x = K["avail_x"]
    pp = avail_x if pod_rank is None else min(pod_rank, avail_x)
    Uxp = Ux_full[:, :pp]
    log("ok",
        f"cOptDMDc(direct): stage 1/4  warm start in {time.time() - t0:.2f}s  "
        f"lm_rank={p}  pod_rank={pp}  F0={F0.shape}  G={G.shape}")

    # ----- 2. eigendecomposition + initial unit-disk projection -----
    log("info", "cOptDMDc(direct): stage 2/4  eigendecomposition + disk projection")
    t = time.time()
    F0_c = F0.astype(np.complex128, copy=False)
    mu, Z = np.linalg.eig(F0_c)
    Z_inv = np.linalg.inv(Z)
    H0 = Ux.T @ X_fit[:, :-1]
    H0_T = H0.T.astype(np.complex128, copy=True)
    alpha     = Z_inv @ H0[:, 0].astype(np.complex128)
    beta_star = Z_inv @ G.astype(np.complex128)

    lam_raw = mu.copy()
    n_unstable = int((np.abs(lam_raw) > 1.0).sum())
    lam = _disk_project(lam_raw)
    log("ok",
        f"cOptDMDc(direct): stage 2/4  eig done in {time.time() - t:.2f}s  "
        f"spectral_radius_raw={np.max(np.abs(lam_raw)):.6f}  "
        f"clipped_unstable={n_unstable}/{p}  "
        f"spectral_radius_proj={np.max(np.abs(lam)):.6f}  H0_T={H0_T.shape}")

    # ----- 3. variable-projection LM (unconstrained + post-step disk projection) -----
    log("info",
        f"cOptDMDc(direct): stage 3/4  LM (unconstrained + disk projection)  "
        f"max_iter={max_iter}  tol={tol}  tol_rel={tol_rel:.1e}  "
        f"patience={patience}  nu0={nu0}  incr={incr}  decr={decr}  "
        f"gmax={gmax}")
    t1 = time.time()

    def fit(l):
        Psi, dPsi = _build_psi_and_dpsi_direct(l, alpha, beta_star, U_fit)
        Up, sp, Vhp = np.linalg.svd(Psi, full_matrices=False)
        Omega = (Vhp.conj().T / sp) @ (Up.conj().T @ H0_T)
        R = H0_T - Psi @ Omega
        return Psi, dPsi, Up, sp, Vhp, Omega, R

    Psi, dPsi, Up, sp, Vhp, Omega, R = fit(lam)
    rho = R.reshape(-1)
    res_norm = np.linalg.norm(rho)
    res_initial = res_norm
    log("info",
        f"cOptDMDc(direct): iter   0  residual={res_norm:.6e}  "
        f"Psi={Psi.shape}  Omega={Omega.shape}")

    stopper = EarlyStopper(tol_abs=tol, tol_rel=tol_rel, patience=patience)
    stop_reason = "max_iter"
    nu = nu0
    iters_done = 0
    total_inner_attempts = 0
    while iters_done < max_iter:
        J = _jacobian_columns(Psi, dPsi, Up, sp, Vhp, Omega, R, p)
        scales = np.linalg.norm(J, axis=0).real

        accepted = False
        inner_restarts = 0
        for inner in range(gmax + 1):
            delta = _solve_lm_unconstrained(J, rho, nu, scales)
            # Unconstrained LM step, then snap back to the unit disk.
            # When the step is small (warm-started case) the projection
            # is a no-op for most λ; only the few that try to leak past
            # |λ| = 1 get clipped.
            lam_trial = _disk_project(lam - delta)
            Psi_t, dPsi_t, Up_t, sp_t, Vhp_t, Omega_t, R_t = fit(lam_trial)
            rho_t = R_t.reshape(-1)
            res_t = np.linalg.norm(rho_t)
            total_inner_attempts += 1
            if res_t <= res_norm:
                accepted = True
                break
            inner_restarts += 1
            nu *= incr
            log("warn",
                f"cOptDMDc(direct): iter {iters_done + 1:3d}.{inner_restarts:<2d}  "
                f"residual rose ({res_t:.6e})  nu={nu:.3e}  "
                f"restarts={inner_restarts}/{gmax}")
        if not accepted:
            log("warn",
                f"cOptDMDc(direct): gmax={gmax} consecutive restarts exhausted "
                f"at outer iter {iters_done + 1}; stopping LM")
            stop_reason = f"gmax exhausted ({gmax} consecutive rejections)"
            break

        lam, Psi, dPsi, Up, sp, Vhp, Omega, R, rho, res_norm = (
            lam_trial, Psi_t, dPsi_t, Up_t, sp_t, Vhp_t,
            Omega_t, R_t, rho_t, res_t)
        iters_done += 1
        log("info",
            f"cOptDMDc(direct): iter {iters_done:3d}  residual={res_norm:.6e}  "
            f"nu={nu:.3e}  ||delta||={np.linalg.norm(delta):.3e}  "
            f"inner_restarts={inner_restarts}  "
            f"spectral_radius={np.max(np.abs(lam)):.6f}")
        if inner_restarts == 0:
            nu = nu / decr
        if stopper.update(res_norm):
            log("ok",
                f"cOptDMDc(direct): early-stopped at iter {iters_done} - "
                f"{stopper.reason}")
            stop_reason = stopper.reason
            break

    log("ok",
        f"cOptDMDc(direct): stage 3/4  LM done in {time.time() - t1:.2f}s  "
        f"accepted_iters={iters_done}  inner_attempts={total_inner_attempts}  "
        f"final_residual={res_norm:.6e}  stop={stop_reason}")

    # ----- 4. reduced operator + forecast -----
    log("info",
        "cOptDMDc(direct): stage 4/4  assembling F_new + in-sample lift + forecast")
    t = time.time()
    F_new = Z @ np.diag(lam) @ Z_inv
    log("ok",
        f"cOptDMDc(direct): stage 4/4  spectral_radius_new={np.max(np.abs(lam)):.6f}")

    log("info",
        f"cOptDMDc(direct): in-sample POD lift  "
        f"X[:, {fit_start_idx}:{fit_split}] -> Uxp Uxp^T X_fit  (pod_rank={pp})")
    X_pred = np.empty_like(X)
    X_pred[:, fit_start_idx:fit_split] = Uxp @ (Uxp.T @ X_fit)
    if fit_start_idx > 0:
        X_pred[:, :fit_start_idx] = Uxp @ (Uxp.T @ X[:, :fit_start_idx])

    n_iters_total = total_steps - fit_steps
    log("info",
        f"cOptDMDc(direct): forecasting {n_iters_total} reduced-space iterations  "
        f"sampling {len(steps) - fit_split} predictions  lift_via_Ux (lm_rank={p})")
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
    log("ok",
        f"cOptDMDc(direct): forecast iterated {iters} steps, sampled "
        f"{target_idx} predictions in {time.time() - tF:.2f}s")

    return {"X_pred":              X_pred.astype(np.float32, copy=False),
            "A":                   F_new,
            "eigenvalues":         lam,
            "rank":                p,
            "pod_rank":            pp,
            "fit_split":           fit_split,
            "lm_initial_residual": float(res_initial),
            "lm_final_residual":   float(res_norm),
            "lm_iters":            iters_done,
            "lm_stop_reason":      stop_reason}
