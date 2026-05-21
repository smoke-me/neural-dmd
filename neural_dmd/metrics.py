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

import numpy as np
import torch
import torch.nn.functional as F

from .params import load_params


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
#
# eval_at is the workhorse: one forward pass per checkpoint accumulates
# both the test loss and the full confusion matrix, from which accuracy
# / precision / recall / F1 are derived in O(C^2). loss_at and metric_at
# are thin wrappers preserved for callers that only need one number.

@torch.no_grad()
def eval_at(model, vec, loader, device, *,
            loss_fn, num_classes: int) -> dict:
    """Single forward pass over `loader` at parameter vector `vec`.
    Returns:
      loss              : float, sample-weighted mean
      confusion_matrix  : (C, C) int64, rows = true class, cols = pred
    Downstream callers derive accuracy/precision/recall/F1 from the
    confusion matrix via metrics_from_confusion()."""
    load_params(model, vec)
    model.eval()
    total_loss = 0.0
    n = 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * y.size(0)
        n += y.size(0)
        pred = logits.argmax(dim=1)
        # cm[true, pred] += 1, vectorised via np.add.at
        yi = y.long().cpu().numpy()
        pi = pred.long().cpu().numpy()
        np.add.at(cm, (yi, pi), 1)
    return {
        "loss":             total_loss / max(n, 1),
        "confusion_matrix": cm,
    }


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """Derive accuracy + macro precision / recall / F1 from a confusion
    matrix `cm[true, pred]`. Macro averages skip classes with zero
    support (recall) or zero predictions (precision); F1 uses the
    intersection.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp        = np.diag(cm)
    support   = cm.sum(axis=1)   # true counts per class
    predicted = cm.sum(axis=0)   # predicted counts per class
    total     = cm.sum()
    accuracy  = float(tp.sum() / total) if total > 0 else float("nan")

    with np.errstate(invalid="ignore", divide="ignore"):
        precision_per = np.where(predicted > 0, tp / np.maximum(predicted, 1), np.nan)
        recall_per    = np.where(support   > 0, tp / np.maximum(support,   1), np.nan)
        denom         = precision_per + recall_per
        f1_per        = np.where(denom > 0, 2 * precision_per * recall_per /
                                  np.where(denom > 0, denom, 1), np.nan)

    def _macro(arr):
        m = np.isfinite(arr)
        return float(arr[m].mean()) if m.any() else float("nan")

    return {
        "accuracy":        accuracy,
        "precision_macro": _macro(precision_per),
        "recall_macro":    _macro(recall_per),
        "f1_macro":        _macro(f1_per),
    }


@torch.no_grad()
def loss_at(model, vec, loader, device, loss_fn) -> float:
    # Implements equation 2: L_k = L(x_k; D). Thin wrapper over eval_at
    # (num_classes inferred from the last linear layer's out_features).
    nc = _infer_num_classes(model)
    return eval_at(model, vec, loader, device,
                   loss_fn=loss_fn, num_classes=nc)["loss"]


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


def _infer_num_classes(model) -> int:
    # Last torch.nn.Linear's out_features; works for our MLP.
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is None:
        raise RuntimeError("metrics: no Linear layer found to infer num_classes")
    return int(last.out_features)
