"""
Rank-selector registry. A rank selector inspects the loaded snapshot
package and decides what truncation rank the DMDc-family methods should
use, patching neural_dmd.config.METHOD_PARAMS in place so opt/copt/dmdc
all see the chosen rank.

Selectors currently registered:

  "fixed"          No-op. Returns None. METHOD_PARAMS[method]["rank"]
                   stays whatever the user set in config. Use this when
                   you want to drive each method with a hand-picked rank.

  "scan"           Held-out forecast-error scan over a candidate list
                   (the historical neural_dmd.rank_scan path, driven by
                   LM_RANK_AUTO_*). Picks the rank with lowest mean
                   relative L2 forecast error on a validation tail.

  "gavish_donoho"  Optimal hard SVD threshold from
                       Gavish & Donoho 2014,
                       "The Optimal Hard Threshold for Singular Values
                       is 4/sqrt(3)".
                   Computes the threshold from the X_fit spectrum and
                   patches the resulting rank into METHOD_PARAMS. No
                   forecast loop is run, so this is orders of magnitude
                   cheaper than "scan" - one SVD of X_fit only.

Adding a selector:
    1. Define a callable `(snap: dict) -> int | None` below. Returning
       None means "don't override; let METHOD_PARAMS keep its rank".
    2. Register it in RANK_SELECTORS under a short name.
    3. Set config.RANK_SELECTOR = "<name>".
    4. (Optional) write diagnostics via experiments.write_metrics_bulk
       under a "__<name>__" key so summary.html can surface them.

Selector signature: (snap: dict) -> int | None
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import config as C
from . import experiments as E
from .gavish_donoho import optimal_threshold
from .log import banner, log, log_spectrum


_LM_METHODS = ("optdmdc", "coptdmdc", "optdmdc_direct", "coptdmdc_direct")


def _patch_method_rank(method: str, rank: int) -> None:
    params = C.METHOD_PARAMS.get(method)
    if params is None:
        return
    old = params.get("rank")
    params["rank"] = int(rank)
    log("ok",
        f"rank selector:  METHOD_PARAMS['{method}']['rank']  "
        f"{old} -> {rank}")


# -----------------------------------------------------------------------
# selectors
# -----------------------------------------------------------------------

def _fixed(snap: dict) -> int | None:
    """No-op: keep the static METHOD_PARAMS rank for every method."""
    log("info",
        "rank selector: 'fixed' - keeping METHOD_PARAMS ranks as configured")
    return None


def _scan(snap: dict) -> int | None:
    """Held-out forecast-error scan. Delegates to neural_dmd.rank_scan
    (the historical LM_RANK_AUTO path)."""
    from .rank_scan import auto_select_lm_rank
    # auto_select_lm_rank reads LM_RANK_AUTO; flip it on so a user who
    # selected this strategy via RANK_SELECTOR doesn't also need to set
    # the legacy flag. Restored after the call.
    prev = getattr(C, "LM_RANK_AUTO", False)
    C.LM_RANK_AUTO = True
    try:
        return auto_select_lm_rank()
    finally:
        C.LM_RANK_AUTO = prev


def _gavish_donoho(snap: dict) -> int | None:
    """Apply the Gavish-Donoho optimal hard threshold to the spectrum
    of X_fit. Returns the integer rank and writes a __gavish_donoho__
    diagnostic block to metrics.json.

    Uses the unknown-noise (median-based) estimator by default. When
    config.GD_USE_KNOWN_SIGMA is True and config.NOISE_SIGMA > 0 the
    known-sigma branch is used instead, with sigma = NOISE_SIGMA scaled
    by the trajectory std (matches how snapshots.inject_snapshot_noise
    interprets NOISE_SIGMA at record time).
    """
    dtype = np.dtype(C.PRECISION)
    X = snap["X"].astype(dtype, copy=False)
    fit_split     = int(snap["fit_split"])
    fit_start_idx = int(snap.get("fit_start_idx", 0))
    X_fit = X[:, fit_start_idx:fit_split]
    n, m_fit = X_fit.shape

    use_known = bool(getattr(C, "GD_USE_KNOWN_SIGMA", False))
    sigma_arg: float | None = None
    if use_known:
        sigma_rel = float(getattr(C, "NOISE_SIGMA", 0.0))
        if sigma_rel > 0.0:
            traj_std = float(np.std(X))
            sigma_arg = sigma_rel * traj_std
            log("info",
                f"gavish-donoho: using KNOWN sigma  sigma_rel={sigma_rel:.3e} "
                f"trajectory_std={traj_std:.3e}  -> sigma_abs={sigma_arg:.3e}")
        else:
            log("info",
                "gavish-donoho: GD_USE_KNOWN_SIGMA=True but NOISE_SIGMA<=0; "
                "falling back to unknown-sigma (median) estimator")

    banner("gavish-donoho threshold",
           X_fit_shape=f"({n},{m_fit})",
           regime=("known-sigma" if sigma_arg is not None else "unknown-sigma"))

    t = time.time()
    s = np.linalg.svd(X_fit, full_matrices=False, compute_uv=False)
    log("ok",
        f"gavish-donoho: SVD(X_fit) done in {time.time() - t:.2f}s  "
        f"spectrum_len={len(s)}")
    log_spectrum("gavish-donoho: spectrum(X_fit)", s)

    res = optimal_threshold(s, shape=(n, m_fit), sigma=sigma_arg)

    # Headline log line - everything a reader needs to verify the
    # threshold choice without scrolling up.
    if res.sigma_known:
        log("ok",
            f"gavish-donoho:  beta={res.beta:.4f}  "
            f"lambda*={res.lambda_star:.4f}  sigma={res.sigma_hat:.3e}  "
            f"threshold={res.threshold:.3e}  rank={res.rank}")
    else:
        log("ok",
            f"gavish-donoho:  beta={res.beta:.4f}  "
            f"lambda*={res.lambda_star:.4f}  omega={res.omega:.4f}  "
            f"sigma_hat=median(s)={res.sigma_hat:.3e}  "
            f"threshold={res.threshold:.3e}  rank={res.rank}")

    # Diagnostic: how far above/below the threshold the boundary
    # singular values sit, so a reader can sanity-check the cut.
    if res.rank > 0 and res.rank < len(s):
        log("info",
            f"gavish-donoho:  boundary  "
            f"s[{res.rank - 1}]={s[res.rank - 1]:.3e}  (last KEPT)  "
            f"s[{res.rank}]={s[res.rank]:.3e}  (first DROPPED)  "
            f"gap_ratio={s[res.rank - 1] / max(s[res.rank], 1e-30):.2f}x")
    elif res.rank == 0:
        log("warn",
            "gavish-donoho: rank=0  every singular value is below the "
            "threshold; the spectrum looks like pure noise. Falling back "
            "to rank=1 to keep the pipeline functional.")
        res.rank = 1
    else:
        log("info",
            "gavish-donoho: rank == full available rank; threshold did "
            "not exclude any singular value (low noise / strong signal).")

    chosen = max(1, int(res.rank))

    for method in _LM_METHODS:
        _patch_method_rank(method, chosen)

    E.write_metrics_bulk("__gavish_donoho__", {
        "rank":              int(chosen),
        "threshold":         float(res.threshold),
        "beta":              float(res.beta),
        "lambda_star":       float(res.lambda_star),
        "omega":             float(res.omega),
        "sigma_hat":         float(res.sigma_hat),
        "sigma_known":       bool(res.sigma_known),
        "n":                 int(n),
        "m_fit":             int(m_fit),
        "spectrum_len":      int(len(s)),
        "s_max":             float(s[0])  if len(s) else 0.0,
        "s_min":             float(s[-1]) if len(s) else 0.0,
        "s_at_boundary":     (float(s[chosen - 1])
                              if 0 < chosen <= len(s) else None),
        "s_first_dropped":   (float(s[chosen])
                              if chosen < len(s) else None),
    })

    return chosen


RANK_SELECTORS: dict[str, Callable[[dict], "int | None"]] = {
    "fixed":         _fixed,
    "scan":          _scan,
    "gavish_donoho": _gavish_donoho,
}


def apply(snap: dict) -> int | None:
    """Dispatch on config.RANK_SELECTOR. Returns the chosen rank, or
    None when the selector is a no-op."""
    name = str(getattr(C, "RANK_SELECTOR", "fixed"))
    if name not in RANK_SELECTORS:
        raise KeyError(
            f"unknown RANK_SELECTOR '{name}'. Registered: {list(RANK_SELECTORS)}")
    return RANK_SELECTORS[name](snap)
