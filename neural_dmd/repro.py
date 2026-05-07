"""
Reproducibility tier enforcement.

`config.REPRO_TIER` selects the trade-off:
    "fast"     - multi-threaded BLAS + GPU training (default)
    "analysis" - multi-threaded training, single-threaded analysis
    "strict"   - single-threaded everywhere + force CPU

Public surface:
    apply_for_training()   call early in scripts/train.py
    apply_for_analysis()   call at the start of runners.run_full_pipeline
    effective_device()     resolve C.DEVICE under the active tier

The thread limits are applied at runtime via threadpoolctl so we don't
have to mess with environment variables before numpy import. The
threadpoolctl context handle is kept module-global so the limits
persist for the rest of the process - threadpool_limits() called
without `with` returns a controller you can call .restore() on later
if needed; we just hold the reference and let it live until exit.
"""

from __future__ import annotations

import os
from typing import Optional

from . import config as C
from .log import log


_VALID_TIERS = ("fast", "analysis", "strict")
_blas_limit_handle = None        # threadpoolctl returns a controller; kept alive


def _tier() -> str:
    t = str(getattr(C, "REPRO_TIER", "fast")).lower()
    if t not in _VALID_TIERS:
        log("warn", f"REPRO_TIER='{t}' invalid; valid={_VALID_TIERS}; using 'fast'")
        return "fast"
    return t


def effective_device() -> str:
    """Resolve C.DEVICE under the active reproducibility tier.
    'strict' forces CPU even when CUDA is available."""
    if _tier() == "strict":
        return "cpu"
    return getattr(C, "DEVICE", "cpu")


def _set_thread_limit_to_one() -> None:
    """Force every thread pool to one worker. Idempotent."""
    global _blas_limit_handle
    try:
        from threadpoolctl import threadpool_limits
        if _blas_limit_handle is None:
            _blas_limit_handle = threadpool_limits(limits=1)
            log("info", "repro: BLAS / OpenMP thread pools limited to 1 (threadpoolctl)")
    except ImportError:
        # Fallback: env vars (only effective if not yet read).
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            os.environ[v] = "1"
        log("warn",
            "repro: threadpoolctl not installed; falling back to env vars "
            "(may be ineffective if numpy/scipy already loaded)")
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        log("info", "repro: torch.set_num_threads(1)")
    except Exception as e:
        log("warn", f"repro: torch thread limit failed ({e})")


def apply_for_training() -> None:
    """Called once near the top of scripts/train.py.
    'strict' forces single-threaded; 'fast' / 'analysis' are no-ops here
    (training stays multi-threaded)."""
    t = _tier()
    log("info", f"repro: tier='{t}' (training stage)")
    if t == "strict":
        _set_thread_limit_to_one()
        log("info", "repro: 'strict' - training will run single-threaded on CPU")


def apply_for_analysis() -> None:
    """Called at the start of runners.run_full_pipeline. 'analysis' and
    'strict' both lock thread pools to 1 from this point on. 'fast' is
    a no-op."""
    t = _tier()
    log("info", f"repro: tier='{t}' (analysis stage)")
    if t in ("analysis", "strict"):
        _set_thread_limit_to_one()


def thread_limit_active() -> bool:
    return _blas_limit_handle is not None
