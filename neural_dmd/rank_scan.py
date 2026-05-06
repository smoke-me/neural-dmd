"""
Auto rank-selection for OptDMDc / cOptDMDc.

Walks a list of candidate LM ranks from low to high. For each rank,
fits a DMDc warm start on the *inner* portion of the fit window and
forecasts across a held-out *validation* segment (the last
`LM_RANK_AUTO_VAL_FRAC` of the fit window). The chosen rank is the one
with lowest mean relative L2 error on the validation segment.

Early stop: after `LM_RANK_AUTO_PATIENCE` consecutive candidates fail
to improve on the running best, the scan exits and that best rank is
locked in for both `optdmdc` and `coptdmdc`.

Why warm-start (no LM) for the scan?
  - The full LM loop is the expensive part; running it at every
    candidate would make the scan minutes per rank.
  - The DMDc warm start is what OptDMDc/cOptDMDc actually start from;
    LM moves eigenvalues by a small amount on top. Forecast quality of
    the warm start is a reasonable proxy for forecast quality of the
    fully-optimised model.
  - The cached SVDs (dmdc._SVD_CACHE) are reused across ranks, so the
    expensive O(n*m^2) factorisation runs once for the whole scan; per
    candidate is just F/G assembly + a forecast loop.

Public surface:
    auto_select_lm_rank() -> int | None
        Returns the chosen rank, or None when LM_RANK_AUTO is off /
        no candidate improved.

Config knobs (all in neural_dmd.config):
    LM_RANK_AUTO              bool   -- master switch (default False)
    LM_RANK_AUTO_CANDIDATES   tuple  -- ranks to try, ascending
    LM_RANK_AUTO_PATIENCE     int    -- consecutive non-improvements to stop
    LM_RANK_AUTO_VAL_FRAC     float  -- fraction of fit window held out
    LM_RANK_AUTO_MIN_IMPROV   float  -- min relative improvement to count
                                       as "better" (defaults 1e-3 = 0.1%)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import config as C
from . import experiments as E
from . import dmdc
from .log import banner, log
from .snapshots import Recorder


def _evaluate_rank(snap: dict, rank: int, *, val_frac: float) -> float:
    """Mean relative L2 forecast error on the held-out validation
    segment when the DMDc warm start is fitted at the given rank."""
    dtype = np.dtype(C.PRECISION)
    X = snap["X"].astype(dtype, copy=False)
    U = snap["U"].astype(dtype, copy=False)
    steps = snap["steps"]
    fit_split     = int(snap["fit_split"])
    fit_start_idx = int(snap.get("fit_start_idx", 0))

    n_fit_snaps = max(fit_split - fit_start_idx, 4)
    n_inner = max(3, int(n_fit_snaps * (1.0 - val_frac)))
    inner_end_idx = fit_start_idx + n_inner

    X_inner = X[:, fit_start_idx:inner_end_idx]
    U_inner = U[:, steps[fit_start_idx:inner_end_idx - 1]]
    n, _ = X.shape
    q = U.shape[0]

    K = dmdc._kernel(X_inner, U_inner,
                     rank=rank, n=n, q=q,
                     m_fit=inner_end_idx - fit_start_idx)
    Ux = K["Ux"]; F = K["F"]; G = K["G"]

    # Forecast: roll forward from the last inner snapshot, sample at
    # each validation snapshot, accumulate mean relative L2 error.
    y = Ux.T @ X[:, inner_end_idx - 1]
    cur_step = int(steps[inner_end_idx - 1])
    target_idx = inner_end_idx
    err_sum = 0.0
    n_compared = 0
    while target_idx < fit_split and cur_step < int(steps[fit_split - 1]):
        y = F @ y + G @ U[:, cur_step]
        cur_step += 1
        if cur_step == int(steps[target_idx]):
            x_pred   = Ux @ y
            x_actual = X[:, target_idx]
            err = (np.linalg.norm(x_pred - x_actual)
                   / max(float(np.linalg.norm(x_actual)), 1e-30))
            err_sum += float(err)
            n_compared += 1
            target_idx += 1
    return err_sum / max(n_compared, 1)


def auto_select_lm_rank() -> int | None:
    """Run the rank scan if config.LM_RANK_AUTO is set, otherwise
    no-op. Returns the chosen rank (or None if off / unsuccessful)."""
    if not bool(getattr(C, "LM_RANK_AUTO", False)):
        return None

    candidates = tuple(getattr(C, "LM_RANK_AUTO_CANDIDATES",
                               (10, 25, 50, 100, 200, 400)))
    patience    = int(getattr(C, "LM_RANK_AUTO_PATIENCE", 2))
    val_frac    = float(getattr(C, "LM_RANK_AUTO_VAL_FRAC", 0.1))
    min_improv  = float(getattr(C, "LM_RANK_AUTO_MIN_IMPROV", 1e-3))

    snap_path = E.snapshots_path()
    if not snap_path.exists():
        log("warn", f"rank scan: snapshots not found at {snap_path}; skipping")
        return None
    snap = Recorder.load(snap_path)

    banner("rank scan",
           candidates=",".join(str(c) for c in candidates),
           patience=patience, val_frac=val_frac,
           min_improv=min_improv)

    history: list[tuple[int, float]] = []
    best_rank: int | None = None
    best_err = float("inf")
    stalled = 0

    for r in candidates:
        t = time.time()
        try:
            err = _evaluate_rank(snap, rank=r, val_frac=val_frac)
        except Exception as e:
            log("warn", f"rank scan: rank={r} failed ({e}); skipping")
            continue
        history.append((int(r), float(err)))
        log("info",
            f"rank scan:  r={r:5d}  val_rel_err={err:.6e}  "
            f"({time.time() - t:.1f}s)")

        if err < best_err * (1.0 - min_improv):
            best_err = err
            best_rank = int(r)
            stalled = 0
        else:
            stalled += 1
            log("info",
                f"rank scan:  no improvement (stalled {stalled}/{patience})")
            if stalled >= patience:
                log("ok",
                    f"rank scan:  early stop after rank={r}; best so far "
                    f"= {best_rank}  (val_err={best_err:.6e})")
                break

    if best_rank is None:
        log("warn", "rank scan: no candidate improved over inf; using "
                   f"smallest candidate ({candidates[0]})")
        best_rank = int(candidates[0])

    # Patch METHOD_PARAMS in place so opt/copt see the chosen rank.
    for method in ("optdmdc", "coptdmdc"):
        params = C.METHOD_PARAMS.get(method)
        if params is not None:
            old = params.get("rank")
            params["rank"] = int(best_rank)
            log("ok",
                f"rank scan:  METHOD_PARAMS['{method}']['rank']  "
                f"{old} -> {best_rank}")

    # Persist scan diagnostics to the experiment's metrics.json.
    E.write_metrics_bulk("__rank_scan__", {
        "best_rank":        int(best_rank),
        "best_val_err":     float(best_err) if best_err != float("inf") else None,
        "candidates":       list(candidates),
        "history":          [{"rank": r, "val_rel_err": e} for r, e in history],
        "patience":         patience,
        "val_frac":         val_frac,
        "min_improv":       min_improv,
    })

    return best_rank
