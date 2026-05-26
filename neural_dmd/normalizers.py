"""
Snapshot normalizer registry.

Motivation: during long training runs, parameter tensors can drift in
overall *magnitude* (a slow zoom-in/out in weight space) while the
*direction* of the weight vector stays roughly fixed. DMD sees this
magnitude drift and burns rank modelling a trivial scaling mode that
carries no information about how the network's behaviour evolves.

The fix is to fit DMD on a unit-norm copy of every recorded snapshot so
the trajectory the operator learns tracks only the *shape* of the
weights over time, not their size. Critically, the snapshots stored on
disk stay UN-normalized: the eval / plot pipeline always sees genuine
trained weights, and DMD-method predictions are de-normalized back to
the original scale before they reach eval / plot. Snapshot normalization
is therefore an ANALYSIS-TIME pre-/post-processing step around each DMD
method, not a record-time mutation - flipping config.SNAPSHOT_NORM does
NOT invalidate the snapshot data_hash.

Adding a new normalizer:
    1. Define a callable `compute_scales(X, sizes) -> np.ndarray` that
       returns a (n_tensors, m_snapshots) matrix of per-tensor scales
       (e.g. ||tensor_i_at_snapshot_j||_2). Return all-ones to make a
       no-op.
    2. Register it in NORMALIZERS under a short name.
    3. Set config.SNAPSHOT_NORM = "<name>" to use it.

The shared `apply_scales(X, scales, sizes, *, inverse)` divides (or
multiplies, for `inverse=True`) every tensor block of every column by
the corresponding scale. Zero scales are treated as 1.0 to avoid 0/0.

Public surface:
    compute_scales(X, sizes, name) -> (n_tensors, m) ndarray
    apply_scales(X, scales, sizes, *, inverse=False) -> ndarray
    NORMALIZERS: dict[str, Callable[[X, sizes], scales]]
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def _scales_off(X: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """No-op normalizer: every scale is 1.0, so divide / multiply are
    identities. Matches the legacy behaviour exactly."""
    return np.ones((len(sizes), X.shape[1]), dtype=X.dtype)


def _scales_per_tensor(X: np.ndarray, sizes: Sequence[int]) -> np.ndarray:
    """Per-tensor L2 norm at every snapshot: scales[i, j] = ||X[block_i, j]||_2.
    Each weight matrix and each bias vector is its own normalization
    group. Zero norms (degenerate / never-updated tensors) pass through
    as 0 here and are clamped to 1.0 by apply_scales so the block is
    left untouched."""
    n_tensors = len(sizes)
    m = X.shape[1]
    out = np.empty((n_tensors, m), dtype=X.dtype)
    offset = 0
    for i, sz in enumerate(sizes):
        block = X[offset:offset + int(sz), :]                 # (sz, m)
        out[i] = np.linalg.norm(block, axis=0)
        offset += int(sz)
    return out


NORMALIZERS: dict[str, Callable[[np.ndarray, Sequence[int]], np.ndarray]] = {
    "off":        _scales_off,
    "per_tensor": _scales_per_tensor,
}


def compute_scales(X: np.ndarray, sizes: Sequence[int], name: str) -> np.ndarray:
    """Look up `name` in NORMALIZERS and return its (n_tensors, m)
    scale matrix for the snapshot stack X."""
    if name not in NORMALIZERS:
        raise KeyError(
            f"unknown normalizer '{name}'. Registered: {list(NORMALIZERS)}")
    if int(sum(sizes)) != int(X.shape[0]):
        raise ValueError(
            f"sizes must sum to X.shape[0] (got sum={sum(sizes)}, "
            f"X.shape[0]={X.shape[0]})")
    return NORMALIZERS[name](X, sizes)


def apply_scales(X: np.ndarray, scales: np.ndarray, sizes: Sequence[int],
                 *, inverse: bool = False) -> np.ndarray:
    """Divide (or multiply if `inverse=True`) every tensor block of
    every column of X by the corresponding scale. Returns a NEW array
    with the same shape and dtype as X.

    Shapes:
        X       : (n, m)   flat-vector snapshot stack
        scales  : (n_tensors, m)   per-tensor per-snapshot scale
        sizes   : (n_tensors,) sum == n

    A scale of 0 is treated as 1.0 in the divide branch so degenerate
    blocks (e.g. zero bias that never updated) pass through unchanged.
    """
    if int(sum(sizes)) != int(X.shape[0]):
        raise ValueError(
            f"sizes must sum to X.shape[0] (got sum={sum(sizes)}, "
            f"X.shape[0]={X.shape[0]})")
    if scales.shape != (len(sizes), X.shape[1]):
        raise ValueError(
            f"scales shape mismatch: got {scales.shape}, "
            f"expected ({len(sizes)}, {X.shape[1]})")

    X_out = X.copy()
    offset = 0
    for i, sz in enumerate(sizes):
        sz = int(sz)
        s = scales[i].astype(X_out.dtype, copy=False)            # (m,)
        if inverse:
            X_out[offset:offset + sz, :] = X_out[offset:offset + sz, :] * s[None, :]
        else:
            safe_s = np.where(s > 0, s, np.ones_like(s))
            X_out[offset:offset + sz, :] = X_out[offset:offset + sz, :] / safe_s[None, :]
        offset += sz
    return X_out
