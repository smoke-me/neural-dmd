"""
cOptDMDc - Constrained Optimized DMD with Control (Rains, Wang, House,
Kaminsky, Tison; J. Comput. Phys. 496 (2024) 112604, sect. 2.4).

Same variable-projection setup as optdmdc.run, with two stability
modifications:

  1. Initial radial projection of gamma onto Re(gamma) <= 0 (paper
     Algorithm 1, line 5):
         gamma <- min(Re(gamma), 0) + Im(gamma) * i
     i.e. any unstable continuous-time eigenvalue is shifted onto the
     imaginary axis (= radially projected onto the unit circle in the
     discrete-time domain).

  2. Linear inequality constraint on the LM update so the step preserves
     stability (paper Eq. 33):
         Re(-delta^(i)) <= Re(-gamma^(i))   <=>   Re(delta) >= Re(gamma)
     hence Re(gamma_new) = Re(gamma) - Re(delta) <= 0.
     The complex constrained LSQ subproblem is recast in real arithmetic
     (paper Eq. 35-36) and solved by scipy.optimize.lsq_linear (TRF), the
     scipy analogue of MATLAB's lsqlin.

Public surface (mirrors dmdc.run):
    run(snap, *, rank=None, pod_rank=None,
        max_iter=50, tol=1e-6, gmax=50, incr=1.5,
        decr=2.0, nu0=2.0, dt=1.0)
        rank      - LM operator rank (sets Jacobian size; memory-bounded).
        pod_rank  - POD basis rank used ONLY for the in-sample
                    reconstruction X_pred[:, :fit_split] = Uxp Uxp^T X_fit;
                    decoupled from `rank` so the in-fit accuracy plot can
                    show ~100% even when the LM operates at low rank.
                    None -> full available POD rank.
        Returns {'X_pred': (n, m) f32, 'A': (p, p) c128,
                 'eigenvalues': (p,) c128, 'gamma': (p,) c128,
                 'rank': int, 'pod_rank': int, 'fit_split': int}.
"""

import time

import numpy as np
from scipy.optimize import lsq_linear

from . import config as C
from .dmdc import _kernel
from .log import log
from .optdmdc import (EarlyStopper, _build_psi_and_dpsi,
                      _jacobian_columns, _real_block_pack)


def _radial_project(gamma):
    # Paper Alg. 1 line 5: any gamma with Re > 0 is mapped to the
    # imaginary axis. In discrete time exp(gamma * dt) has |.| > 1 when
    # Re(gamma) > 0, so this projects unstable modes onto the unit circle.
    return np.where(gamma.real > 0, 1j * gamma.imag, gamma)


def _solve_lm_constrained(J, rho, gamma, nu, scales):
    """LM subproblem under the linear inequality Re(delta) >= Re(gamma).
    Real-block recast (Eq. 35-36): variables are stacked [a; b] with
    a = Re(delta), b = Im(delta). The constraint becomes a >= Re(gamma)
    on the first r entries; b is unconstrained. Solved via
    scipy.optimize.lsq_linear with bounds.
    """
    big_A, big_b = _real_block_pack(J, rho, nu, scales)
    r = J.shape[1]
    lb = np.concatenate([gamma.real, np.full(r, -np.inf)])
    ub = np.full(2 * r,  np.inf)
    res = lsq_linear(big_A, big_b, bounds=(lb, ub),
                     method="trf", max_iter=200, verbose=0)
    return res.x[:r] + 1j * res.x[r:]


def run(snap: dict, *, rank: int | None = None, pod_rank: int | None = None,
        max_iter: int = 50, tol: float = 1e-6, gmax: int = 50,
        incr: float = 1.5, decr: float = 2.0, nu0: float = 2.0,
        dt: float = 1.0,
        tol_rel: float = 1e-3, patience: int = 3) -> dict:
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
        f"cOptDMDc: stage 1/4  fit data assembled  X_fit={X_fit.shape}  "
        f"U_fit={U_fit.shape}  n={n}  q={q}  rank={rank}  pod_rank={pod_rank}")
    t0 = time.time()

    # ----- 1. warm-start via shared dmdc kernel -----
    log("info", "cOptDMDc: stage 1/4  warm-starting via DMDc kernel")
    K = _kernel(X_fit, U_fit, rank=rank, n=n, q=q, m_fit=fit_split)
    Ux = K["Ux"]; F0 = K["F"]; G = K["G"]; p = K["p"]
    Ux_full = K["Ux_full"]; avail_x = K["avail_x"]
    pp = avail_x if pod_rank is None else min(pod_rank, avail_x)
    Uxp = Ux_full[:, :pp]
    log("ok",
        f"cOptDMDc: stage 1/4  warm start in {time.time() - t0:.2f}s  "
        f"lm_rank={p}  pod_rank={pp}  F0={F0.shape}  G={G.shape}")

    # ----- 2. eigendecomposition + initial radial projection -----
    # F0 is small (p x p); promote to complex128 so eig/log/exp stay
    # at full precision even when bulk arrays are f32.
    log("info", "cOptDMDc: stage 2/4  eigendecomposition + radial projection")
    t = time.time()
    F0_c = F0.astype(np.complex128, copy=False)
    mu, Z = np.linalg.eig(F0_c)
    Z_inv = np.linalg.inv(Z)
    H0 = Ux.T @ X_fit[:, :-1]
    H0_T = H0.T.astype(np.complex128, copy=True)
    alpha     = Z_inv @ H0[:, 0].astype(np.complex128)
    beta_star = Z_inv @ G.astype(np.complex128)

    gamma_raw = np.log(mu) / dt
    n_unstable = int((gamma_raw.real > 0).sum())
    gamma = _radial_project(gamma_raw)
    log("ok",
        f"cOptDMDc: stage 2/4  eig done in {time.time() - t:.2f}s  "
        f"spectral_radius_raw={np.max(np.abs(mu)):.6f}  "
        f"clipped_unstable={n_unstable}/{p}  "
        f"max_real_gamma={gamma.real.max():.3e}  H0_T={H0_T.shape}")

    # ----- 3. variable-projection LM (constrained Re(gamma) <= 0) -----
    log("info",
        f"cOptDMDc: stage 3/4  constrained LM  "
        f"max_iter={max_iter}  tol={tol}  tol_rel={tol_rel:.1e}  "
        f"patience={patience}  nu0={nu0}  incr={incr}  decr={decr}  "
        f"gmax={gmax}")
    t1 = time.time()

    def fit(g):
        Psi, dPsi = _build_psi_and_dpsi(g, alpha, beta_star, U_fit, dt)
        Up, sp, Vhp = np.linalg.svd(Psi, full_matrices=False)
        Omega = (Vhp.conj().T / sp) @ (Up.conj().T @ H0_T)
        R = H0_T - Psi @ Omega
        return Psi, dPsi, Up, sp, Vhp, Omega, R

    Psi, dPsi, Up, sp, Vhp, Omega, R = fit(gamma)
    rho = R.reshape(-1)
    res_norm = np.linalg.norm(rho)
    res_initial = res_norm
    log("info",
        f"cOptDMDc: iter   0  residual={res_norm:.6e}  "
        f"Psi={Psi.shape}  Omega={Omega.shape}")

    stopper = EarlyStopper(tol_abs=tol, tol_rel=tol_rel, patience=patience)
    stop_reason = "max_iter"
    nu = nu0
    restarts = 0
    iters_done = 0
    for it in range(1, max_iter + 1):
        iters_done = it
        J = _jacobian_columns(Psi, dPsi, Up, sp, Vhp, Omega, R, p)
        scales = np.linalg.norm(J, axis=0).real

        delta = _solve_lm_constrained(J, rho, gamma, nu, scales)
        # Re(delta) >= Re(gamma) is the LM constraint, so Re(gamma_new) <= 0
        # is automatic; the radial project below is a numerical safety net
        # for solver round-off.
        gamma_trial = _radial_project(gamma - delta)

        Psi_t, dPsi_t, Up_t, sp_t, Vhp_t, Omega_t, R_t = fit(gamma_trial)
        rho_t = R_t.reshape(-1)
        res_t = np.linalg.norm(rho_t)

        if res_t > res_norm:
            restarts += 1
            nu *= incr
            log("warn",
                f"cOptDMDc: iter {it:3d}  residual rose ({res_t:.6e})  "
                f"nu={nu:.3e}  restarts={restarts}/{gmax}")
            if restarts > gmax:
                log("warn", "cOptDMDc: gmax exceeded, stopping LM loop")
                stop_reason = f"gmax exceeded ({restarts}>{gmax})"
                break
            continue
        gamma, Psi, dPsi, Up, sp, Vhp, Omega, R, rho, res_norm = (
            gamma_trial, Psi_t, dPsi_t, Up_t, sp_t, Vhp_t,
            Omega_t, R_t, rho_t, res_t)
        log("info",
            f"cOptDMDc: iter {it:3d}  residual={res_norm:.6e}  nu={nu:.3e}  "
            f"||delta||={np.linalg.norm(delta):.3e}  "
            f"max_real_gamma={gamma.real.max():.3e}")
        if restarts == 0:
            nu = nu / decr
        restarts = 0
        if stopper.update(res_norm):
            log("ok", f"cOptDMDc: early-stopped at iter {it} - {stopper.reason}")
            stop_reason = stopper.reason
            break

    log("ok",
        f"cOptDMDc: stage 3/4  LM done in {time.time() - t1:.2f}s  "
        f"final_residual={res_norm:.6e}  stop={stop_reason}")

    # ----- 4. reduced operator + forecast -----
    log("info", "cOptDMDc: stage 4/4  assembling F_new + in-sample lift + forecast")
    t = time.time()
    mu_new = np.exp(gamma * dt)
    F_new = Z @ np.diag(mu_new) @ Z_inv
    log("ok",
        f"cOptDMDc: stage 4/4  spectral_radius_new={np.max(np.abs(mu_new)):.6f}  "
        f"max_real_gamma={gamma.real.max():.3e}")

    log("info",
        f"cOptDMDc: in-sample POD lift  "
        f"X[:, {fit_start_idx}:{fit_split}] -> Uxp Uxp^T X_fit  (pod_rank={pp})")
    X_pred = np.empty_like(X)
    X_pred[:, fit_start_idx:fit_split] = Uxp @ (Uxp.T @ X_fit)
    if fit_start_idx > 0:
        X_pred[:, :fit_start_idx] = Uxp @ (Uxp.T @ X[:, :fit_start_idx])

    n_iters_total = total_steps - fit_steps
    log("info",
        f"cOptDMDc: forecasting {n_iters_total} reduced-space iterations  "
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
        f"cOptDMDc: forecast iterated {iters} steps, sampled "
        f"{target_idx} predictions in {time.time() - tF:.2f}s")

    return {"X_pred":             X_pred.astype(np.float32, copy=False),
            "A":                  F_new,
            "eigenvalues":        mu_new,
            "gamma":              gamma,
            "rank":               p,
            "pod_rank":           pp,
            "fit_split":          fit_split,
            "lm_initial_residual": float(res_initial),
            "lm_final_residual":   float(res_norm),
            "lm_iters":           iters_done,
            "lm_stop_reason":     stop_reason}
