"""
Optimizer registry. Single source of truth for which torch optimizers
the training loop can build.

Adding a new optimizer:
    1. Define a builder below: (params, lr, **kwargs) -> torch.optim.Optimizer.
       The builder swallows any unknown kwargs in **kwargs so config can
       carry extras without breaking other optimizers.
    2. Register it in OPTIMIZERS under a short name.
    3. (Optional) Add defaults to config.OPTIMIZER_PARAMS[<name>].
    4. Set config.OPTIMIZER = "<name>" to use it.

Builder signature: (params, lr: float, **kwargs) -> torch.optim.Optimizer
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch


def _adam(params: Iterable, lr: float, **kw) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        params,
        lr=lr,
        betas=kw.get("betas", (0.9, 0.999)),
        eps=kw.get("eps", 1e-8),
        weight_decay=kw.get("weight_decay", 0.0),
    )


def _sgd(params: Iterable, lr: float, **kw) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=kw.get("momentum", 0.0),
        dampening=kw.get("dampening", 0.0),
        weight_decay=kw.get("weight_decay", 0.0),
        nesterov=kw.get("nesterov", False),
    )


OPTIMIZERS: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "adam": _adam,
    "sgd":  _sgd,
}


def build(name: str, params: Iterable, lr: float,
          extra: dict | None = None) -> torch.optim.Optimizer:
    """Look up `name` in OPTIMIZERS and instantiate it. `extra` is the
    per-optimizer kwargs dict (e.g. config.OPTIMIZER_PARAMS[name])."""
    if name not in OPTIMIZERS:
        raise KeyError(
            f"unknown optimizer '{name}'. Registered: {list(OPTIMIZERS)}")
    return OPTIMIZERS[name](params, lr, **(extra or {}))
