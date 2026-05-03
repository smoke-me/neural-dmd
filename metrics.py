"""
Loss functions, per-batch metrics, and helpers that evaluate them on a
parameter vector.

To plug in a new loss / metric:
  1. Define a callable below.
  2. Register it in LOSSES or METRICS under a short name.
  3. Set config.LOSS or config.METRIC to that name.

Loss signature  : (logits, targets) -> scalar tensor (mean reduction)
Metric signature: (logits, targets) -> python float in [0, 1]
"""

import torch
import torch.nn.functional as F

from params import load_params


# --- loss functions ---

def cross_entropy(logits, y):
    return F.cross_entropy(logits, y)


# --- per-batch metrics ---

def classification_accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


# Registries. config.LOSS / config.METRIC names look up here.
LOSSES = {
    "cross_entropy": cross_entropy,
}

METRICS = {
    "accuracy": classification_accuracy,
}


# --- evaluators that take a parameter vector ---

@torch.no_grad()
def loss_at(model, vec, loader, device, loss_fn) -> float:
    # Implements equation 2: L_k = L(x_k; D).
    # Loads the parameter vector x_k into `model`, runs a forward pass over
    # the entire `loader`, and returns the mean loss. The per-batch losses
    # are weighted by batch size so an uneven last batch is handled correctly.
    load_params(model, vec)
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += loss_fn(model(x), y).item() * y.size(0)
        n += y.size(0)
    return total / n


@torch.no_grad()
def metric_at(model, vec, loader, device, metric_fn) -> float:
    # Same as loss_at but for a per-batch metric (e.g. accuracy).
    load_params(model, vec)
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += metric_fn(model(x), y) * y.size(0)
        n += y.size(0)
    return total / n
