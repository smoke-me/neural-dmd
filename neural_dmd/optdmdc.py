"""
OptDMDc - Optimized DMD with Control (Askham-Kutz variable projection
extended with exogenous inputs; sect. 2.3 of Rains et al., JCP 2024).

Builds on dmdc._kernel for the warm start, then refines the reduced
eigenvalues via Levenberg-Marquardt variable projection on the
exponential data-fitting residual

    H_0^T = Psi(gamma, t) * Omega,   Omega = Psi^+ H_0^T

and minimises  || H_0^T - Psi(gamma) Omega(gamma) ||_F  in gamma alone.
NO stability constraint - eigenvalues are free to land anywhere in the
complex plane. Use coptdmdc.run for the constrained variant.

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

Module-private (also imported by coptdmdc.py - shared math):
    _build_psi_and_dpsi(gamma, alpha, beta_star, U_fit, dt)
    _jacobian_columns(Psi, dPsi, Up, sp, Vhp, Omega, R, p)
    _real_block_pack(J, rho, nu, scales) -> (big_A, big_b)
"""

import time

import numpy as np

from .dmdc import _kernel
from .log import log


# ---------------------------------------------------------------------------
# shared math helpers (also reused by coptdmdc.py)
# ---------------------------------------------------------------------------

def _build_psi_and_dpsi(gamma, alpha, beta_star, U_fit, dt):
    """Per-iteration Psi (mm, r) and dPsi (mm, r) where dPsi[:, ell] is
    dPsi/dgamma[ell]. The full D_ell matrix from paper Eq. (31) has only
    its ell-th column nonzero - that column equals dPsi[:, ell] - so we
    return the dense (mm, r) form and the caller assembles D_ell columns
    in _jacobian_columns.

    Closed-form on uniform time grid (t[k] = k * dt) via cumsum:
        Psi[k, i] = alpha[i] * exp(gamma[i] * t[k])
                   + sum_{j=0..k-1} c[i, j] * exp(gamma[i] * t[k-1-j])
        c[i, j]   = beta_star[i, :] @ U_fit[:, j]
    On a uniform grid exp(gamma * t[k-1-j]) = exp(gamma * t[k-1]) *
    exp(-gamma * t[j]), so the inner sum collapses to a cumulative sum.
    """
    r = gamma.shape[0]
    _, mm = U_fit.shape
    t = np.arange(mm) * dt

    auto = alpha[None, :] * np.exp(np.outer(t, gamma))           # (mm, r)
    c = beta_star @ U_fit                                        # (r, mm)
    eg = np.exp(-np.outer(gamma, t))                             # (r, mm)
    inner   = c * eg
    inner_t = inner * t[None, :]
    csumS = np.cumsum(inner,   axis=1)
    csumT = np.cumsum(inner_t, axis=1)
    S = np.zeros_like(csumS); S[:, 1:] = csumS[:, :-1]
    T = np.zeros_like(csumT); T[:, 1:] = csumT[:, :-1]

    ctrl  = np.zeros_like(auto)
    dctrl = np.zeros_like(auto)
    if mm > 1:
        shift = np.exp(np.outer(t[:mm - 1], gamma))              # (mm-1, r)
        ctrl[1:, :]  = shift * S[:, 1:].T
        dctrl[1:, :] = shift * (t[:mm - 1, None] * S[:, 1:].T - T[:, 1:].T)

    Psi  = auto + ctrl
    dPsi = t[:, None] * auto + dctrl
    return Psi, dPsi


def _jacobian_columns(Psi, dPsi, Up, sp, Vhp, Omega, R, p):
    """Build J (mm*r, p) complex by Eq. (32):
        J^mat_j = -(I - U_psi U_psi^*) D_j Omega
                  - U_psi Sigma_psi^{-1} V_psi^* (D_j^* R)
    D_j has only its j-th column nonzero with values dPsi[:, j], so:
        - (I - UU^*) D_j Omega         -> outer(proj_dPsi[:, j], Omega[j, :])
        - U Sigma^-1 V^* (D_j^* R)     -> outer(u_vec_j, w_j)
          with u_vec_j = U_psi @ (V_psi^*[:, j] / sp), w_j = dPsi[:, j].conj() @ R.
    Per-column cost is O(mm * r); total is O(mm * r * p).
    """
    mm, r = dPsi.shape
    proj_dPsi = dPsi - Up @ (Up.conj().T @ dPsi)                 # (mm, r)
    u_vec_all = Up @ (Vhp / sp[:, None])                         # (mm, r)
    w_all     = dPsi.conj().T @ R                                # (r, r)

    J = np.empty((mm * r, p), dtype=complex)
    for j in range(p):
        block = (-np.outer(proj_dPsi[:, j], Omega[j, :])
                 - np.outer(u_vec_all[:, j], w_all[j, :]))
        J[:, j] = block.reshape(-1)
    return J


def _real_block_pack(J, rho, nu, scales):
    """Real-block recast of the complex augmented LSQ system
        min || [J; nu * diag(scales)] delta - [rho; 0] ||_2^2
    with J = S + Ti, delta = a + bi:
        | S  -T |[a]   |Re(rho)|
        | T   S |[b] = |Im(rho)|
        |nuM  0 |     |   0   |
        | 0  nuM|     |   0   |
    Returns the real (2M + 2r) x 2r matrix big_A and rhs vector big_b.
    Caller solves by numpy.linalg.lstsq (unconstrained, optdmdc) or by
    scipy.optimize.lsq_linear with bounds (constrained, coptdmdc).
    """
    M = J.shape[0]; r = J.shape[1]
    S = J.real; Tm = J.imag
    nuM = np.diag(nu * scales)
    big_A = np.zeros((2 * M + 2 * r, 2 * r))
    big_A[:M,            :r ] =  S
    big_A[:M,            r: ] = -Tm
    big_A[M:2 * M,       :r ] =  Tm
    big_A[M:2 * M,       r: ] =  S
    big_A[2 * M:2 * M + r, :r] = nuM
    big_A[2 * M + r:,      r:] = nuM
    big_b = np.concatenate([rho.real, rho.imag, np.zeros(2 * r)])
    return big_A, big_b


def _solve_lm_unconstrained(J, rho, nu, scales):
    big_A, big_b = _real_block_pack(J, rho, nu, scales)
    sol, *_ = np.linalg.lstsq(big_A, big_b, rcond=None)
    r = J.shape[1]
    return sol[:r] + 1j * sol[r:]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def run(snap: dict, *, rank: int | None = None, pod_rank: int | None = None,
        max_iter: int = 50, tol: float = 1e-6, gmax: int = 50,
        incr: float = 1.5, decr: float = 2.0, nu0: float = 2.0,
        dt: float = 1.0) -> dict:
    X = snap["X"].astype(np.float64, copy=False)
    U = snap["U"].astype(np.float64, copy=False)
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
        f"OptDMDc: stage 1/4  fit data assembled  X_fit={X_fit.shape}  "
        f"U_fit={U_fit.shape}  n={n}  q={q}  rank={rank}  pod_rank={pod_rank}")
    t0 = time.time()

    # ----- 1. warm-start via shared dmdc kernel -----
    log("info", "OptDMDc: stage 1/4  warm-starting via DMDc kernel")
    K = _kernel(X_fit, U_fit, rank=rank, n=n, q=q, m_fit=fit_split)
    Ux = K["Ux"]; F0 = K["F"]; G = K["G"]; p = K["p"]
    Ux_full = K["Ux_full"]; avail_x = K["avail_x"]
    pp = avail_x if pod_rank is None else min(pod_rank, avail_x)
    Uxp = Ux_full[:, :pp]                                        # POD basis for in-fit lift
    log("ok",
        f"OptDMDc: stage 1/4  warm start in {time.time() - t0:.2f}s  "
        f"lm_rank={p}  pod_rank={pp}  F0={F0.shape}  G={G.shape}")

    # ----- 2. eigendecomposition + initial gamma -----
    log("info", "OptDMDc: stage 2/4  eigendecomposition + initial gamma")
    t = time.time()
    mu, Z = np.linalg.eig(F0)                                    # F0 = Z diag(mu) Z^-1
    Z_inv = np.linalg.inv(Z)
    H0 = Ux.T @ X_fit[:, :-1]                                    # (p, m_fit-1)
    H0_T = H0.T.astype(complex, copy=True)                       # (m_fit-1, p)
    alpha     = Z_inv @ H0[:, 0]                                 # (p,)
    beta_star = Z_inv @ G                                        # (p, q)
    gamma = np.log(mu).astype(complex) / dt
    log("ok",
        f"OptDMDc: stage 2/4  eig done in {time.time() - t:.2f}s  "
        f"spectral_radius={np.max(np.abs(mu)):.6f}  "
        f"max_real_gamma={gamma.real.max():.3e}  H0_T={H0_T.shape}")

    # ----- 3. variable-projection LM (unconstrained) -----
    log("info",
        f"OptDMDc: stage 3/4  variable-projection LM  "
        f"max_iter={max_iter}  tol={tol}  nu0={nu0}  incr={incr}  decr={decr}  "
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
    log("info",
        f"OptDMDc: iter   0  residual={res_norm:.6e}  "
        f"Psi={Psi.shape}  Omega={Omega.shape}")

    nu = nu0
    restarts = 0
    for it in range(1, max_iter + 1):
        J = _jacobian_columns(Psi, dPsi, Up, sp, Vhp, Omega, R, p)
        scales = np.linalg.norm(J, axis=0).real

        delta = _solve_lm_unconstrained(J, rho, nu, scales)
        gamma_trial = gamma - delta

        Psi_t, dPsi_t, Up_t, sp_t, Vhp_t, Omega_t, R_t = fit(gamma_trial)
        rho_t = R_t.reshape(-1)
        res_t = np.linalg.norm(rho_t)

        if res_t > res_norm:
            restarts += 1
            nu *= incr
            log("warn",
                f"OptDMDc: iter {it:3d}  residual rose ({res_t:.6e})  "
                f"nu={nu:.3e}  restarts={restarts}/{gmax}")
            if restarts > gmax:
                log("warn", "OptDMDc: gmax exceeded, stopping LM loop")
                break
            continue
        gamma, Psi, dPsi, Up, sp, Vhp, Omega, R, rho, res_norm = (
            gamma_trial, Psi_t, dPsi_t, Up_t, sp_t, Vhp_t,
            Omega_t, R_t, rho_t, res_t)
        log("info",
            f"OptDMDc: iter {it:3d}  residual={res_norm:.6e}  nu={nu:.3e}  "
            f"||delta||={np.linalg.norm(delta):.3e}")
        if restarts == 0:
            nu = nu / decr
        restarts = 0
        if res_norm < tol:
            log("ok", f"OptDMDc: converged at iter {it} (residual<{tol:.1e})")
            break

    log("ok",
        f"OptDMDc: stage 3/4  LM done in {time.time() - t1:.2f}s  "
        f"final_residual={res_norm:.6e}")

    # ----- 4. reduced operator + forecast -----
    log("info", "OptDMDc: stage 4/4  assembling F_new + in-sample lift + forecast")
    t = time.time()
    mu_new = np.exp(gamma * dt)
    F_new = Z @ np.diag(mu_new) @ Z_inv                          # (p, p) complex
    log("ok",
        f"OptDMDc: stage 4/4  spectral_radius_new={np.max(np.abs(mu_new)):.6f}  "
        f"max_real_gamma={gamma.real.max():.3e}")

    # In-sample reconstruction lifts via the (possibly higher-rank) POD
    # basis Uxp so the in-fit accuracy plot reflects the POD-truncation
    # bound at pod_rank, not the lower lm_rank used for dynamics.
    log("info",
        f"OptDMDc: in-sample POD lift  "
        f"X[:, {fit_start_idx}:{fit_split}] -> Uxp Uxp^T X_fit  (pod_rank={pp})")
    X_pred = np.empty_like(X)
    X_pred[:, fit_start_idx:fit_split] = Uxp @ (Uxp.T @ X_fit)
    if fit_start_idx > 0:
        X_pred[:, :fit_start_idx] = Uxp @ (Uxp.T @ X[:, :fit_start_idx])

    # Forecast in (complex) reduced space; lift Ux @ y and take real part
    # since the underlying state x is real in our application. Conjugate
    # symmetry of mu/Z is not enforced - if the LM step pushes gamma off
    # conjugate pairs, F_new becomes general complex but Re(Ux @ y) is the
    # right projection back to R^n.
    n_iters_total = total_steps - fit_steps
    log("info",
        f"OptDMDc: forecasting {n_iters_total} reduced-space iterations  "
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
        f"OptDMDc: forecast iterated {iters} steps, sampled "
        f"{target_idx} predictions in {time.time() - tF:.2f}s")

    return {"X_pred":      X_pred.astype(np.float32, copy=False),
            "A":           F_new,
            "eigenvalues": mu_new,
            "gamma":       gamma,
            "rank":        p,
            "pod_rank":    pp,
            "fit_split":   fit_split}
