"""
Control-input registry for the DMDc dynamical model.

DMDc fits a linear model of the parameter-vector trajectory under a
per-step control signal:

    x_{k+1} = A x_k + B u_k

This module owns u_k. The control is composed by concatenating the
outputs of a registered list of `ControlSource` instances. Each source
contributes a fixed-dim chunk; the composer concatenates them in the
order declared by config.CONTROLS. Changing the list re-defines u_k -
no DMDc method code needs to change because all of them treat U as an
opaque (q, total_steps-1) matrix.

# Why include a low-rank batch summary in u_k

The gradient step that drives x_k -> x_{k+1} depends on (i) the LR and
(ii) the batch B_k of training examples processed that step. The LR is
already a scalar control. The batch is the other half of the
information that determines the update direction: feeding a few
projection coefficients of B_k onto a fixed low-rank basis exposes
"what kind of data drove this step" to DMDc, so its linear B can map
batch content to weight-update direction. Because the basis is fixed
across all steps, the coordinates live in a consistent space and DMDc
can learn a meaningful coefficient matrix.

# Registry pattern

Each source has:
    - a string `name`
    - `setup(setup_ctx)` for one-time precomputation (e.g. PCA basis)
    - `dim` property: the number of scalars it appends to u_k
    - `step(model, optimizer, step, batch)` -> 1-D ndarray of length `dim`

Adding a new source:
    1. Subclass `ControlSource` and implement setup / dim / step.
    2. Register in CONTROLS under a short name.
    3. Append the name to config.CONTROLS to enable it for a run.
    4. (Optional) hyperparameters go in config.CONTROL_PARAMS[name].

# Composer signature

The composer's step method matches the callable interface that
neural_dmd.snapshots.Recorder expects:

    composer.step(model, optimizer, *, step, batch) -> np.ndarray (q,)

Public surface:
    CONTROLS:        dict[str, type[ControlSource]]
    build_controls(names, params, setup_ctx) -> ControlComposer
    ControlComposer: .dim, .names_dims, .step(...)
"""

from __future__ import annotations

import abc
from typing import Iterable

import numpy as np
import torch

from .log import log


class ControlSource(abc.ABC):
    """One contributor to the per-step control vector u_k."""

    name: str = "abstract"

    def __init__(self, **params):
        self.params = dict(params)

    @abc.abstractmethod
    def setup(self, *, setup_ctx: dict) -> None:
        """One-time precomputation (e.g. PCA basis). Called once before
        the training loop starts. `setup_ctx` carries dataset name,
        data root, batch size, seed, device etc."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Number of scalars this source contributes to every u_k."""

    @abc.abstractmethod
    def step(self, *, model, optimizer, step: int, batch) -> np.ndarray:
        """Compute this source's slice of u_k (shape (dim,))."""


# ---------------------------------------------------------------------------
# concrete sources
# ---------------------------------------------------------------------------

class LRControl(ControlSource):
    """Single-scalar control: the optimizer's current learning rate.
    The historical default; preserved so flipping CONTROLS = ("lr",)
    reproduces the legacy behaviour exactly."""

    name = "lr"

    def setup(self, *, setup_ctx: dict) -> None:
        pass

    @property
    def dim(self) -> int:
        return 1

    def step(self, *, model, optimizer, step: int, batch) -> np.ndarray:
        return np.array(
            [float(optimizer.param_groups[0]["lr"])],
            dtype=np.float32,
        )


class BatchPCAControl(ControlSource):
    """Low-rank summary of the current training batch.

    On setup, computes the top-k principal directions of the FLATTENED
    training set (one-time cost, ~seconds for MNIST). At each step,
    projects the batch's mean image onto those k directions to produce
    a k-scalar coordinate in a fixed PCA frame.

    Why batch-mean projection (vs per-image projections):
        DMDc operates on a fixed q-dim control. Per-image projections
        would give batch_size * k scalars - too many, and the ordering
        within the batch is arbitrary. The batch mean is a stable
        sufficient statistic for the average gradient signal that drove
        the step; its projection lives in a consistent k-dim space
        across all steps and across re-runs.

    Why a fixed dataset-wide basis (vs per-batch SVD):
        DMDc's B matrix needs control coordinates whose meaning is the
        same in every column of U; otherwise B has no fixed semantics.
        Per-batch SVD would change basis per step and confound this.
    """

    name = "batch_pca"

    def __init__(self, k: int = 8, **kw):
        super().__init__(k=k, **kw)
        self.k = int(k)
        self.mean: np.ndarray | None = None          # (D,)
        self.V:    np.ndarray | None = None          # (k, D)
        self.eigvals: np.ndarray | None = None       # (k,)

    def setup(self, *, setup_ctx: dict) -> None:
        from .data import get_loaders

        dataset_name = setup_ctx["dataset_name"]
        data_root    = setup_ctx["data_root"]
        seed         = int(setup_ctx.get("seed", 0))
        pca_batch    = int(setup_ctx.get("pca_setup_batch", 2000))

        # Walk the training set once to stack flattened images. Cheap on
        # MNIST (60000 * 784 * 4B ≈ 190 MB). Use a larger batch than the
        # training batch size to amortise DataLoader overhead.
        loader, _ = get_loaders(dataset_name, batch_size=pca_batch,
                                eval_batch=pca_batch, root=data_root,
                                seed=seed)
        chunks: list[np.ndarray] = []
        for x, _ in loader:
            chunks.append(
                x.view(x.shape[0], -1).cpu().numpy().astype(np.float32))
        X = np.concatenate(chunks, axis=0)                   # (N, D)
        N, D = X.shape
        self.mean = X.mean(axis=0)                           # (D,)
        Xc = X - self.mean[None, :]

        # Eigendecompose the (small) D x D Gram. For D=784 this is a
        # ~10 ms call; far cheaper than np.linalg.svd on the (N, D)
        # data matrix.
        cov = (Xc.T @ Xc) / max(N - 1, 1)                    # (D, D)
        vals, vecs = np.linalg.eigh(cov)                     # ascending
        order = np.argsort(-vals)[:self.k]                   # top-k by |λ|
        self.V       = vecs[:, order].T.astype(np.float32)   # (k, D)
        self.eigvals = vals[order].astype(np.float32)        # (k,)

        log("info",
            f"BatchPCAControl.setup: dataset={dataset_name}  N={N}  D={D}  "
            f"k={self.k}  top_eigvals={self.eigvals.tolist()}")

    @property
    def dim(self) -> int:
        return self.k

    def step(self, *, model, optimizer, step: int, batch) -> np.ndarray:
        if batch is None:
            # No batch handed in -> return zeros. Keeps the pipeline
            # alive when a caller forgets to thread the batch through;
            # logs once at most via the Recorder.
            return np.zeros(self.k, dtype=np.float32)
        x, _ = batch
        x_flat = x.view(x.shape[0], -1).detach().cpu().numpy().astype(np.float32)
        mean_img = x_flat.mean(axis=0)                       # (D,)
        centered = mean_img - self.mean                      # (D,)
        return (self.V @ centered).astype(np.float32)        # (k,)


# ---------------------------------------------------------------------------
# registry + composer
# ---------------------------------------------------------------------------

CONTROLS: dict[str, type[ControlSource]] = {
    "lr":        LRControl,
    "batch_pca": BatchPCAControl,
}


class ControlComposer:
    """Concatenates per-source contributions into a single u_k. Owned
    by the Recorder via Recorder.control_fn."""

    def __init__(self, sources: list[ControlSource]):
        self.sources = list(sources)
        self._dim = sum(s.dim for s in self.sources)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def names_dims(self) -> list[tuple[str, int]]:
        return [(s.name, s.dim) for s in self.sources]

    def step(self, model, optimizer, *,
             step: int, batch=None) -> np.ndarray:
        parts = [s.step(model=model, optimizer=optimizer,
                        step=step, batch=batch)
                 for s in self.sources]
        return np.concatenate(parts).astype(np.float32)


def build_controls(names: Iterable[str], params: dict[str, dict],
                   *, setup_ctx: dict) -> ControlComposer:
    """Instantiate every control source in `names`, call setup, and
    wrap them in a ControlComposer."""
    sources: list[ControlSource] = []
    for n in names:
        if n not in CONTROLS:
            raise KeyError(
                f"unknown control '{n}'. Registered: {list(CONTROLS)}")
        cls = CONTROLS[n]
        sp  = (params or {}).get(n, {}) or {}
        s   = cls(**sp)
        s.setup(setup_ctx=setup_ctx)
        sources.append(s)
    composer = ControlComposer(sources)
    log("info",
        f"controls: built composer  total_dim={composer.dim}  "
        f"sources={composer.names_dims}")
    return composer
