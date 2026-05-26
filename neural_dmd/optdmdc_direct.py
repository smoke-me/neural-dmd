"""
OptDMDc (direct-λ form) - same variable-projection LM as
neural_dmd.optdmdc, parametrized by the discrete-time eigenvalue λ
directly instead of γ = log(λ)/dt.

Why direct?
    The log-form basis builder evaluates exp(γ · t) and exp(-γ · t) for
    t up to fit-window length. When the warm-start operator has tiny
    eigenvalues (|λ| ≈ 0) the corresponding γ has |Re(γ)| huge; the
    exp() then overflows to +inf, producing NaN in the Jacobian and
    crashing the SVD inside the LM step. The direct form skips the
    log/exp pair entirely: Ψ and ∂Ψ/∂λ are built by iterating the
    scalar linear recurrence
        auto[k+1]  = λ · auto[k],            auto[0]  = α
        drive[k+1] = λ · drive[k] + c[k],    drive[0] = 0
    and applying the product rule for derivatives. λ = 0 ⇒ everything
    collapses to 0 (and to the most recent control input), no overflow.

What we lose vs the log form:
    - For irregular time grids, the analytical closed form via
      cumsum-of-exponentials is cleaner with γ; here we'd need a
      per-step recurrence anyway. We assume a uniform grid (the only
      regime the rest of the codebase records snapshots on).
    - Conjugate-pair preservation is implicit (numpy's eig keeps pairs
      to float-precision on real-valued operators). The LM step takes
      complex δ - pairs stay paired unless rounding splits them, same
      as the γ form.

What we keep:
    - Identical Jacobian formula (variable-projection Eq. 32):
      _jacobian_columns reads dPsi opaque-y, so the function from
      optdmdc.py applies unchanged.
    - Identical LM driver, EarlyStopper, real-block pack, lstsq solver.
    - Same warm-start path (dmdc._kernel).

Public surface mirrors neural_dmd.optdmdc.run.
"""

import time

import numpy as np

from . import config as C
from .dmdc import _kernel
from .log import log
from .optdmdc import (EarlyStopper, _jacobian_columns,
                      _solve_lm_unconstrained)


def _build_psi_and_dpsi_direct(lam: np.ndarray,
                               alpha: np.ndarray,
                               beta_star: np.ndarray,
                               U_fit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Direct-λ analogue of optdmdc._build_psi_and_dpsi.

    Continuous-time formula (log form):
        Psi[k, i] = α_i exp(γ_i t_k)
                  + Σ_{j=0..k-1} c_{i,j} exp(γ_i (t_k - t_{j+1}))
    becomes, on a uniform grid (t_k = k · dt) with λ_i = exp(γ_i · dt):
        Psi[k, i] = α_i λ_i^k
                  + Σ_{j=0..k-1} c_{i,j} λ_i^(k-1-j)

    Both terms are accumulated by forward recurrence (no powering of λ,
    no exp). The autonomous part: auto[k+1] = λ auto[k]. The driven part
    is the scalar control system: drive[k+1] = λ drive[k] + c[k]. Their
    derivatives w.r.t. λ follow by the product rule:
        dauto[k+1]  = auto[k]  + λ dauto[k]
        ddrive[k+1] = drive[k] + λ ddrive[k]

    Numerics: |λ| < 1 chains decay to 0 (no underflow trouble); |λ| > 1
    grows polynomially-in-time-times-geometrically-in-|λ|. Float64
    saturates at |λ|^mm ≈ 1e308; for our mm ≈ 350 fit windows this is
    |λ| ≈ 7 - well above any spectral radius we see in practice.

    Parameters
    ----------
    lam       : (r,) complex128 - discrete-time eigenvalues to fit.
    alpha     : (r,) complex128 - initial reduced state projected onto
                the eigenbasis of the warm-start operator F0.
    beta_star : (r, q) complex128 - control coupling β* = Z^{-1} G.
    U_fit     : (q, mm-1) - control sequence aligned with X_fit columns
                0..mm-2 (the control that drove x_k -> x_{k+1}).

    Returns
    -------
    Psi  : (mm, r) complex128
    dPsi : (mm, r) complex128 - dPsi[k, i] = ∂Psi[k, i] / ∂λ_i
    """
    r = lam.shape[0]
    # mm = U_fit.shape[1] = number of transitions in the fit window =
    # number of rows in H0_T (= m_fit - 1). Psi must match.
    _, mm = U_fit.shape

    # c[i, j] = control coefficient applied to mode i at step j.
    c = (beta_star @ U_fit).astype(np.complex128, copy=False)        # (r, mm)
    lam128 = lam.astype(np.complex128, copy=False)

    Psi  = np.empty((mm, r), dtype=np.complex128)
    dPsi = np.empty((mm, r), dtype=np.complex128)

    auto   = alpha.astype(np.complex128, copy=True)
    drive  = np.zeros(r, dtype=np.complex128)
    dauto  = np.zeros(r, dtype=np.complex128)
    ddrive = np.zeros(r, dtype=np.complex128)

    Psi[0]  = auto + drive                                            # = alpha
    dPsi[0] = dauto + ddrive                                          # = 0

    # Tight inner loop. mm ≈ 350 outer iters, each a few vectorized ops
    # on (r,) arrays - r ≈ 144. Total cost ~2 ms; cheap relative to the
    # SVD that follows in fit().
    for k in range(mm - 1):
        # Order matters: ∂-recursions read the OLD auto / drive before
        # those are overwritten by the primary update.
        dauto  = auto  + lam128 * dauto
        ddrive = drive + lam128 * ddrive
        auto   = lam128 * auto
        drive  = lam128 * drive + c[:, k]
        Psi[k + 1]  = auto + drive
        dPsi[k + 1] = dauto + ddrive

    return Psi, dPsi


def run(snap: dict, *, rank: int | None = None, pod_rank: int | None = None,
        max_iter: int = 50, tol: float = 1e-6, gmax: int = 50,
        incr: float = 1.5, decr: float = 2.0, nu0: float = 2.0,
        dt: float = 1.0,
        tol_rel: float = 1e-3, patience: int = 3) -> dict:
    # `dt` is accepted only for API compatibility with optdmdc.run;
    # the direct form parameterizes by λ which already absorbs dt.
    del dt   # not used

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
        f"OptDMDc(direct): stage 1/4  fit data assembled  X_fit={X_fit.shape}  "
        f"U_fit={U_fit.shape}  n={n}  q={q}  rank={rank}  pod_rank={pod_rank}")
    t0 = time.time()

    # ----- 1. warm-start via shared dmdc kernel -----
    log("info", "OptDMDc(direct): stage 1/4  warm-starting via DMDc kernel")
    K = _kernel(X_fit, U_fit, rank=rank, n=n, q=q, m_fit=fit_split)
    Ux = K["Ux"]; F0 = K["F"]; G = K["G"]; p = K["p"]
    Ux_full = K["Ux_full"]; avail_x = K["avail_x"]
    pp = avail_x if pod_rank is None else min(pod_rank, avail_x)
    Uxp = Ux_full[:, :pp]
    log("ok",
        f"OptDMDc(direct): stage 1/4  warm start in {time.time() - t0:.2f}s  "
        f"lm_rank={p}  pod_rank={pp}  F0={F0.shape}  G={G.shape}")

    # ----- 2. eigendecomposition (initial λ = warm-start eigenvalues) -----
    log("info", "OptDMDc(direct): stage 2/4  eigendecomposition + initial λ")
    t = time.time()
    F0_c = F0.astype(np.complex128, copy=False)
    mu, Z = np.linalg.eig(F0_c)                                # F0 = Z diag(mu) Z^-1
    Z_inv = np.linalg.inv(Z)
    H0 = Ux.T @ X_fit[:, :-1]                                  # (p, mm-1)
    H0_T = H0.T.astype(np.complex128, copy=True)               # (mm-1, p)
    alpha     = Z_inv @ H0[:, 0].astype(np.complex128)         # (p,)
    beta_star = Z_inv @ G.astype(np.complex128)                # (p, q)
    lam = mu.copy()                                            # parameters we fit, complex128
    log("ok",
        f"OptDMDc(direct): stage 2/4  eig done in {time.time() - t:.2f}s  "
        f"spectral_radius={np.max(np.abs(lam)):.6f}  "
        f"min|λ|={np.min(np.abs(lam)):.3e}  H0_T={H0_T.shape}")

    # ----- 3. variable-projection LM (unconstrained, in λ-space) -----
    log("info",
        f"OptDMDc(direct): stage 3/4  variable-projection LM  "
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
        f"OptDMDc(direct): iter   0  residual={res_norm:.6e}  "
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
            lam_trial = lam - delta
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
                f"OptDMDc(direct): iter {iters_done + 1:3d}.{inner_restarts:<2d}  "
                f"residual rose ({res_t:.6e})  nu={nu:.3e}  "
                f"restarts={inner_restarts}/{gmax}")
        if not accepted:
            log("warn",
                f"OptDMDc(direct): gmax={gmax} consecutive restarts exhausted "
                f"at outer iter {iters_done + 1}; stopping LM")
            stop_reason = f"gmax exhausted ({gmax} consecutive rejections)"
            break

        lam, Psi, dPsi, Up, sp, Vhp, Omega, R, rho, res_norm = (
            lam_trial, Psi_t, dPsi_t, Up_t, sp_t, Vhp_t,
            Omega_t, R_t, rho_t, res_t)
        iters_done += 1
        log("info",
            f"OptDMDc(direct): iter {iters_done:3d}  residual={res_norm:.6e}  "
            f"nu={nu:.3e}  ||delta||={np.linalg.norm(delta):.3e}  "
            f"inner_restarts={inner_restarts}  "
            f"spectral_radius={np.max(np.abs(lam)):.6f}")
        if inner_restarts == 0:
            nu = nu / decr
        if stopper.update(res_norm):
            log("ok",
                f"OptDMDc(direct): early-stopped at iter {iters_done} - "
                f"{stopper.reason}")
            stop_reason = stopper.reason
            break

    log("ok",
        f"OptDMDc(direct): stage 3/4  LM done in {time.time() - t1:.2f}s  "
        f"accepted_iters={iters_done}  inner_attempts={total_inner_attempts}  "
        f"final_residual={res_norm:.6e}  stop={stop_reason}")

    # ----- 4. reduced operator + forecast -----
    log("info",
        "OptDMDc(direct): stage 4/4  assembling F_new + in-sample lift + forecast")
    t = time.time()
    F_new = Z @ np.diag(lam) @ Z_inv                                  # (p, p) complex
    log("ok",
        f"OptDMDc(direct): stage 4/4  spectral_radius_new={np.max(np.abs(lam)):.6f}")

    log("info",
        f"OptDMDc(direct): in-sample POD lift  "
        f"X[:, {fit_start_idx}:{fit_split}] -> Uxp Uxp^T X_fit  (pod_rank={pp})")
    X_pred = np.empty_like(X)
    X_pred[:, fit_start_idx:fit_split] = Uxp @ (Uxp.T @ X_fit)
    if fit_start_idx > 0:
        X_pred[:, :fit_start_idx] = Uxp @ (Uxp.T @ X[:, :fit_start_idx])

    n_iters_total = total_steps - fit_steps
    log("info",
        f"OptDMDc(direct): forecasting {n_iters_total} reduced-space iterations  "
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
        f"OptDMDc(direct): forecast iterated {iters} steps, sampled "
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
