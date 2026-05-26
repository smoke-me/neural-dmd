"""
Snapshot normalizer registry.

Motivation: during long training runs, parameter tensors can drift in
overall *magnitude* (a slow zoom-in/out in weight space) while the
*direction* of the weight vector stays roughly fixed. DMD sees this
magnitude drift and burns rank modelling a trivial scaling mode that
carries no information about how the network's behaviour evolves.

The fix is to rescale every recorded checkpoint so the per-tensor norms
are pinned to 1.0; the trajectory then tracks only the shape of the
weights over time, not their size. Both the real network's checkpoints
and the DMD forecast live in this normalized space - the plots that
compare them stay apples-to-apples, but the test-loss / test-accuracy
curves reflect the *normalized* network (each parameter tensor unit-
normed) rather than the raw trained network. When SNAPSHOT_NORM = "off"
(the default), behaviour is identical to the legacy flatten_params path.

Adding a new normalizer:
    1. Define a callable below: `(model) -> np.ndarray` returning the
       flat parameter vector after normalization. The output order MUST
       match neural_dmd.params.flatten_params (concatenation of every
       tensor in model.parameters()) so load_params can reverse it.
    2. Register it in NORMALIZERS under a short name.
    3. Set config.SNAPSHOT_NORM = "<name>" to use it.

Normalizer signature: (model: torch.nn.Module) -> np.ndarray  (float32)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .params import flatten_params


def _off(model: torch.nn.Module) -> np.ndarray:
    """Identity: flatten with no rescaling. Matches the historical
    behaviour exactly when SNAPSHOT_NORM = 'off'."""
    return flatten_params(model)


def _per_tensor(model: torch.nn.Module) -> np.ndarray:
    """Divide every parameter tensor by its own L2 norm before
    concatenation. Zero-norm tensors (rare; e.g. bias initialised at 0
    and never updated) pass through unchanged to avoid 0/0 = NaN.

    Each weight matrix and each bias vector is its own normalization
    group. The output is a unit-multi-norm vector: every component-
    tensor has ||.||_2 = 1 (or 0 for the degenerate case).
    """
    parts: list[np.ndarray] = []
    for p in model.parameters():
        v = p.detach().reshape(-1).cpu().numpy().astype(np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm > 0.0:
            v = v / nrm
        parts.append(v)
    return np.concatenate(parts)


NORMALIZERS: dict[str, Callable[[torch.nn.Module], np.ndarray]] = {
    "off":        _off,
    "per_tensor": _per_tensor,
}


def get(name: str) -> Callable[[torch.nn.Module], np.ndarray]:
    if name not in NORMALIZERS:
        raise KeyError(
            f"unknown normalizer '{name}'. Registered: {list(NORMALIZERS)}")
    return NORMALIZERS[name]
