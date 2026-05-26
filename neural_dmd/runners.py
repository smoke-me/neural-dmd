"""
Per-method analyze + plot dispatch. The thin script wrappers under
scripts/ all delegate here, and run_exp.py uses these directly to
orchestrate full pipelines.

Conventions:

    do_analyze(method)              reads snapshots, runs the method, saves
                                    analysis npz, calls log_eigenvalue_report,
                                    registers metrics in the experiment.
    do_plot_loss(method)            test-loss curve (actual vs predicted)
    do_plot_accuracy(method)        parameter-vector L2-distance accuracy
                                    (DMDc reconstruction quality, NOT
                                    classification accuracy)
    do_plot_classification(method)  test-set classification metrics
                                    (accuracy / precision / recall / F1)
                                    of the predicted network vs the
                                    actual checkpoint network
    do_plot_eigenvalues(method)     |λ| scatter on the unit circle
    do_plot_combined(methods)       cross-method overlay of loss +
                                    classification metrics (one figure)

All file paths are resolved through neural_dmd.experiments. Per-method
hyperparameters come from neural_dmd.config.METHOD_PARAMS[method] and
are forwarded as kwargs to the registered run callable.

# Curve caching

For test-set evaluation we need a forward pass per checkpoint x_k for
both the actual trajectory and each method's predicted trajectory. To
amortise that cost we cache the joint (loss + confusion-matrix-derived
metrics) curves on disk:

    <exp_dir>/actual_eval_curve.npz     shared across methods (actual)
    <exp_dir>/curves/<method>.npz       per method (predicted)

A snapshot-path + idx + X_pred-fingerprint check keeps the cache fresh.
plot_loss, plot_classification and plot_combined all read from these
caches, so re-running them is near-instant.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from . import config as C
from . import experiments as E
from .figures import (
    classification_grid_plot,
    combined_overlay_plot,
    comparison_plot,
    eigenvalue_plot,
)
from .log import banner, log, progress
from .methods import get as get_method
from .snapshots import Recorder, eval_indices


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def do_analyze(method: str) -> None:
    info = get_method(method)
    runner = info["run"]
    label  = info["label"]
    has_lm = info.get("has_lm", False)
    params = dict(C.METHOD_PARAMS.get(method, {}))

    eid = E.get_or_init_exp()
    out_path = E.analysis_path(method)

    banner(f"analyze[{method}] start",
           exp=eid, dataset=C.DATASET,
           snapshots=E.snapshots_path(), output=out_path,
           params=params)

    if not E.snapshots_path().exists():
        raise FileNotFoundError(
            f"snapshots not found at {E.snapshots_path()} -- run train.py first")
    snap = Recorder.load(E.snapshots_path())
    log("info",
        f"loaded {E.snapshots_path()}  X={snap['X'].shape}  "
        f"fit_split={snap['fit_split']}  fit_start_idx={snap['fit_start_idx']}  "
        f"total_steps={snap['total_steps']}")

    t0 = time.time()
    out = runner(snap, **params)
    wall = time.time() - t0
    log("ok", f"{label}: run() finished in {wall:.2f}s")

    eigvals = np.asarray(out.get("eigenvalues",
                                 np.linalg.eigvals(out["A"]))).astype(np.complex128)
    eig_metrics = E.log_eigenvalue_report(label, eigvals)

    payload = {
        "X_pred":      out["X_pred"],
        "eigenvalues": eigvals,
        "rank":        np.int64(out["rank"]),
        "fit_split":   np.int64(out["fit_split"]),
    }
    if "gamma" in out:
        payload["gamma"] = np.asarray(out["gamma"], dtype=np.complex128)
    if "pod_rank" in out:
        payload["pod_rank"] = np.int64(out["pod_rank"])
    np.savez(out_path, **payload)
    log("ok",
        f"saved {out_path}  X_pred={out['X_pred'].shape}  "
        f"eigvals={eigvals.shape}  rank={out['rank']}")

    metrics = {
        "rank":            int(out["rank"]),
        "wall_seconds":    float(wall),
        "n_outside":       eig_metrics["n_outside"],
        "spectral_radius": eig_metrics["spectral_radius"],
        "outside_top":     eig_metrics["outside_top"],
    }
    if "pod_rank" in out:
        metrics["pod_rank"] = int(out["pod_rank"])
    # convergence stats: methods may attach lm_* keys to their return
    for k in ("lm_initial_residual", "lm_final_residual",
              "lm_iters", "lm_restarts", "lm_stop_reason"):
        if k in out:
            metrics[k] = out[k]
    if has_lm and "lm_initial_residual" not in metrics:
        # methods that don't currently report stats - leave NaN slots
        # untouched so summary.md just shows '—'.
        pass
    E.write_metrics_bulk(method, metrics)


# ---------------------------------------------------------------------------
# eval-curve plumbing (loss + classification in one forward pass per
# checkpoint, cached to disk)
# ---------------------------------------------------------------------------

# Metric keys produced by metrics.eval_at + metrics_from_confusion. The
# order is canonical: any code that iterates "all metrics" follows this
# tuple so the summary table columns stay in sync with the curves.
_CURVE_METRICS = ("loss", "accuracy", "precision", "recall", "f1")

def _eval_curve(label, net, X_cols, idx, test_loader, *,
                loss_fn, num_classes):
    """Evaluate the network at every column index in `idx` of X_cols.
    Returns a dict with arrays for each of _CURVE_METRICS and cm_final
    (the confusion matrix at the last grid point). One forward pass per
    grid point - the cost most callers want to amortise via caching."""
    from .metrics import eval_at, metrics_from_confusion

    t = time.time()
    n = len(idx)
    out = {k: np.empty(n, dtype=np.float64) for k in _CURVE_METRICS}
    cm_final = None
    for i, k in enumerate(idx, start=1):
        res = eval_at(net, X_cols[:, k], test_loader, C.DEVICE,
                      loss_fn=loss_fn, num_classes=num_classes)
        cm  = res["confusion_matrix"]
        m   = metrics_from_confusion(cm)
        out["loss"][i - 1]      = res["loss"]
        out["accuracy"][i - 1]  = m["accuracy"]
        out["precision"][i - 1] = m["precision_macro"]
        out["recall"][i - 1]    = m["recall_macro"]
        out["f1"][i - 1]        = m["f1_macro"]
        cm_final = cm
        progress(f"{label} eval curve", i, n, t, every_pct=25.0)
    out["cm_final"] = cm_final.astype(np.int64) \
                      if cm_final is not None else \
                      np.zeros((num_classes, num_classes), dtype=np.int64)
    return out


def _save_curve(path, curve, *, idx, snap_path, fingerprint=None):
    payload = {k: curve[k] for k in (*_CURVE_METRICS, "cm_final")}
    payload["idx"]       = np.asarray(idx)
    payload["snap_path"] = snap_path
    if fingerprint is not None:
        payload["fingerprint"] = fingerprint
    np.savez(path, **payload)


def _load_curve_if_fresh(path, *, idx, snap_path, fingerprint=None):
    """Return the cached curve dict iff the on-disk file matches the
    expected idx + snap_path (+ optional fingerprint). Else None."""
    if not path.exists():
        return None
    try:
        d = np.load(path)
    except Exception as e:
        log("warn", f"curve cache unreadable at {path}: {e}; recomputing")
        return None
    files = set(d.files)
    if "idx" not in files or "snap_path" not in files:
        return None
    if str(d["snap_path"]) != snap_path:
        return None
    if not np.array_equal(d["idx"], np.asarray(idx)):
        return None
    if fingerprint is not None:
        if "fingerprint" not in files or str(d["fingerprint"]) != fingerprint:
            return None
    return {k: d[k] for k in (*_CURVE_METRICS, "cm_final")}


# Module-level cache for the actual eval curve. Identical across every
# method that runs in this experiment (same snapshots, same eval grid,
# same dataset, same loss).
_ACTUAL_EVAL_CACHE: dict[tuple, dict] = {}


def _actual_eval_curve_cached(net, snap, idx, test_loader, loss_fn,
                              num_classes):
    snap_path = str(E.snapshots_path())
    key = (snap_path, tuple(int(i) for i in idx), C.DATASET, C.LOSS)
    if key in _ACTUAL_EVAL_CACHE:
        log("ok", "actual eval curve: in-memory cache HIT")
        return _ACTUAL_EVAL_CACHE[key]

    disk_cache = E.exp_dir() / "actual_eval_curve.npz"
    cached = _load_curve_if_fresh(disk_cache, idx=idx, snap_path=snap_path)
    if cached is not None:
        _ACTUAL_EVAL_CACHE[key] = cached
        log("ok", f"actual eval curve: disk cache HIT ({disk_cache.name})")
        return cached

    log("info",
        "actual eval curve: computing (will be reused for the rest of "
        "this experiment's plot calls)")
    curve = _eval_curve("actual", net, snap["X"], idx, test_loader,
                        loss_fn=loss_fn, num_classes=num_classes)
    _ACTUAL_EVAL_CACHE[key] = curve
    try:
        _save_curve(disk_cache, curve, idx=idx, snap_path=snap_path)
        log("ok", f"actual eval curve: cached to {disk_cache.name}")
    except Exception as e:
        log("warn", f"actual eval curve: failed to write disk cache ({e})")
    return curve


def _pred_eval_curve_cached(method, label, net, X_pred, idx, test_loader,
                            loss_fn, num_classes):
    """Per-method predicted-network curve. Cache key includes a sha256
    fingerprint of X_pred so re-running analyze with new hyperparameters
    invalidates the curve automatically."""
    snap_path  = str(E.snapshots_path())
    curves_dir = E.curves_dir()
    cache_path = curves_dir / f"{method}.npz"
    fp = hashlib.sha256(np.ascontiguousarray(X_pred).tobytes()).hexdigest()[:16]

    cached = _load_curve_if_fresh(cache_path, idx=idx, snap_path=snap_path,
                                  fingerprint=fp)
    if cached is not None:
        log("ok", f"{label} pred eval curve: disk cache HIT ({cache_path.name})")
        return cached

    curve = _eval_curve(label, net, X_pred, idx, test_loader,
                        loss_fn=loss_fn, num_classes=num_classes)
    try:
        _save_curve(cache_path, curve, idx=idx, snap_path=snap_path,
                    fingerprint=fp)
        log("ok", f"{label} pred eval curve: cached to {cache_path.name}")
    except Exception as e:
        log("warn", f"{label} pred eval curve: failed to write disk cache ({e})")
    return curve


def _last(arr) -> float:
    """Last value as a python float; NaN if the array is empty."""
    a = np.asarray(arr, dtype=float)
    return float(a[-1]) if a.size else float("nan")


def _signed_gap(forecast_last: float, real_last: float) -> float:
    """Signed forecast - real at the final step. NaN if either side is
    NaN; positive numbers mean the forecast over-shoots, negative means
    it under-shoots."""
    if not (np.isfinite(forecast_last) and np.isfinite(real_last)):
        return float("nan")
    return float(forecast_last - real_last)


def _write_curve_metrics(method: str, actual: dict, pred: dict) -> None:
    """Stamp per-method final test metrics for the forecasted network +
    its signed gap vs the real network (forecast - real, at the last
    grid point) + the predicted network's confusion matrix at the
    final step. The "best/min/max" bookkeeping was dropped on user
    request - the final value is what matters for forecast quality."""
    payload: dict = {}
    for m in _CURVE_METRICS:
        p_final = _last(pred[m])
        a_final = _last(actual[m])
        payload[f"pred_{m}_final"] = p_final
        payload[f"gap_{m}_final"]  = _signed_gap(p_final, a_final)
    cm = pred.get("cm_final")
    if cm is not None:
        payload["pred_cm_final"] = np.asarray(cm).astype(int).tolist()
    E.write_metrics_bulk(method, payload)


def _write_actual_summary_once(actual: dict) -> None:
    """Stamp the real network's headline test metrics under '__actual__'
    so write_summary can render them once at the top (shared across
    every DMDc method)."""
    payload: dict = {f"{m}_final": _last(actual[m]) for m in _CURVE_METRICS}
    payload["cm_final"] = np.asarray(actual["cm_final"]).astype(int).tolist()
    E.write_metrics_bulk("__actual__", payload)


# ---------------------------------------------------------------------------
# plot helpers
# ---------------------------------------------------------------------------

def _load_analysis(method: str):
    snap = Recorder.load(E.snapshots_path())
    cache = np.load(E.analysis_path(method))
    return snap, cache


def _grid(snap):
    idx = eval_indices(snap["steps"], snap["forecast_every"])
    grid_steps = snap["steps"][idx]
    fit_start_grid = int(np.searchsorted(grid_steps,
                                         snap.get("fit_start_step", 0)))
    plot_split = int(np.searchsorted(grid_steps, snap["fit_steps"]))
    return idx, grid_steps, fit_start_grid, plot_split


def _eval_setup(method: str):
    """Common setup for plot_loss / plot_classification: load snapshots
    + analysis cache, build the eval grid, instantiate the model + test
    loader + loss fn, and compute (or fetch cached) actual + predicted
    eval curves.

    Returns (snap, cache, idx, grid_steps, fit_start_grid, plot_split,
             actual_curve, pred_curve, num_classes, label, rank).
    """
    import torch  # noqa: F401
    from .data import get_loaders
    from .metrics import LOSSES
    from .model import MLP

    info  = get_method(method)
    label = info["label"]

    snap, cache = _load_analysis(method)
    X_pred    = cache["X_pred"]
    fit_split = int(cache["fit_split"])
    rank      = int(cache["rank"])
    log("info",
        f"loaded snapshots ({snap['X'].shape}) + analysis cache  "
        f"rank={rank}  fit_split={fit_split}")

    idx, grid_steps, fit_start_grid, plot_split = _grid(snap)
    log("info",
        f"evaluation grid: {len(idx)} points (every {snap['forecast_every']} steps)  "
        f"fit_start_grid={fit_start_grid}  plot_split={plot_split}")

    _, test_loader = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)
    net = MLP(C.ARCH).to(C.DEVICE)
    loss_fn = LOSSES[C.LOSS]
    num_classes = int(C.ARCH[-1])

    actual = _actual_eval_curve_cached(net, snap, idx, test_loader,
                                       loss_fn, num_classes)
    pred   = _pred_eval_curve_cached(method, label, net, X_pred, idx,
                                     test_loader, loss_fn, num_classes)
    return (snap, cache, idx, grid_steps, fit_start_grid, plot_split,
            actual, pred, num_classes, label, rank)


def do_plot_loss(method: str) -> None:
    eid = E.get_or_init_exp()
    out_path = E.plot_path(method, "loss")

    banner(f"plot_loss[{method}] start",
           exp=eid, dataset=C.DATASET, loss=C.LOSS, device=C.DEVICE,
           output=out_path)

    (snap, cache, idx, grid_steps, fit_start_grid, plot_split,
     actual, pred, _nc, label, rank) = _eval_setup(method)
    fit_split = int(cache["fit_split"])

    summary = comparison_plot(
        grid_steps, actual["loss"], pred["loss"], plot_split,
        fit_start_idx=fit_start_grid,
        out_path=out_path,
        title=f"{label} loss-curve forecast on {C.DATASET.upper()}   "
              f"·   fit on {fit_split - cache.get('fit_start_idx', 0):d} dense snapshots   "
              f"·   eval on {len(idx)} grid points   ·   rank {rank}",
        ylabel=f"test {C.LOSS}",
        actual_label=r"real network   $\mathcal{L}(x_k;\mathcal{D})$",
        pred_label=fr"{label} forecast   $\mathcal{{L}}(\hat{{x}}_k;\mathcal{{D}})$",
        summary_label="mean |Δloss| in fit window = {in_:.4f}",
    )
    log("ok", f"saved {out_path}")

    # plot_classification calls _write_curve_metrics too; the second
    # call is idempotent (same cache, same arithmetic).
    _write_curve_metrics(method, actual, pred)
    _write_actual_summary_once(actual)
    E.write_metrics_bulk(method, {"in_sample_dloss": summary["in_sample"]})


def do_plot_accuracy(method: str) -> None:
    info  = get_method(method)
    label = info["label"]
    eid = E.get_or_init_exp()
    out_path = E.plot_path(method, "accuracy")

    banner(f"plot_accuracy[{method}] start", exp=eid, dataset=C.DATASET,
           output=out_path)

    snap, cache = _load_analysis(method)
    X_pred    = cache["X_pred"]
    fit_split = int(cache["fit_split"])
    rank      = int(cache["rank"])
    log("info",
        f"loaded snapshots ({snap['X'].shape}) + analysis cache  "
        f"rank={rank}  fit_split={fit_split}")

    idx, grid_steps, fit_start_grid, plot_split = _grid(snap)

    Xg   = snap["X"][:, idx]
    Xp_g = X_pred[:, idx]
    e    = np.linalg.norm(Xp_g - Xg, axis=0) / (np.linalg.norm(Xg, axis=0) + 1e-12)
    acc_pred   = 100.0 * (1.0 - e)
    acc_actual = np.full(len(idx), 100.0)

    # Accuracy is bounded above at 100% (actual reference is constant);
    # below 0% means ||x_pred - x|| exceeds ||x|| (the prediction is
    # farther from origin than the truth, often by orders of magnitude
    # when a method blows up). Clip y-axis to a sensible window so the
    # 100% reference + the meaningful negative range stay readable; the
    # plot annotates how many points fell off-scale.
    summary = comparison_plot(
        grid_steps, acc_actual, acc_pred, plot_split,
        fit_start_idx=fit_start_grid,
        y_clip=(-25.0, 105.0),
        out_path=out_path,
        title=f"{label} parameter-prediction accuracy on {C.DATASET.upper()}   "
              f"·   fit on {fit_split - cache.get('fit_start_idx', 0):d} dense snapshots   "
              f"·   eval on {len(idx)} grid points   ·   rank {rank}",
        ylabel="reconstruction accuracy  [%]",
        actual_label=r"real network   $x_k$  (reference, 100%)",
        pred_label=fr"{label} forecast   $100\cdot(1 - \|\hat{{x}}_k - x_k\|/\|x_k\|)$",
        summary_label="mean |Δaccuracy| in fit window = {in_:.4f}%",
    )
    log("ok", f"saved {out_path}")
    E.write_metrics_bulk(method, {"in_sample_dacc_pct": summary["in_sample"]})


def do_plot_classification(method: str) -> None:
    """Test-set classification metrics (accuracy / precision / recall /
    F1, macro) of the actual checkpoint network vs the predicted
    network, over the same eval grid as plot_loss."""
    eid = E.get_or_init_exp()
    out_path = E.plot_path(method, "classification")

    banner(f"plot_classification[{method}] start",
           exp=eid, dataset=C.DATASET, device=C.DEVICE, output=out_path)

    (snap, cache, idx, grid_steps, fit_start_grid, plot_split,
     actual, pred, _nc, label, rank) = _eval_setup(method)
    fit_split = int(cache["fit_split"])

    classification_grid_plot(
        grid_steps, actual, pred, plot_split,
        fit_start_idx=fit_start_grid,
        out_path=out_path,
        title=f"{label} classification metrics on {C.DATASET.upper()}   "
              f"·   fit on {fit_split - cache.get('fit_start_idx', 0):d} dense snapshots   "
              f"·   eval on {len(idx)} grid points   ·   rank {rank}",
        actual_label="real network",
        pred_label=f"{label} forecast",
    )
    log("ok", f"saved {out_path}")

    # plot_loss may already have written these on the same run; the
    # re-stamp is harmless (same arithmetic, same cache).
    _write_curve_metrics(method, actual, pred)
    _write_actual_summary_once(actual)


def do_plot_eigenvalues(method: str) -> None:
    info  = get_method(method)
    label = info["label"]
    eid = E.get_or_init_exp()
    out_path = E.plot_path(method, "eigenvalues")

    banner(f"plot_eigenvalues[{method}] start", exp=eid, dataset=C.DATASET,
           output=out_path)

    cache = np.load(E.analysis_path(method))
    eigvals = cache["eigenvalues"]
    rank    = int(cache["rank"])
    mags    = np.abs(eigvals)
    spectral_radius = float(mags.max()) if mags.size else 0.0
    tol = C.EIG_STABLE_TOL

    log("info", f"loaded {len(eigvals)} eigenvalues  rank={rank}")
    log("info",
        f"  inside  unit circle (|λ|≤1+{tol:.0e}): "
        f"{int((mags <= 1.0 + tol).sum())}/{len(eigvals)}")
    log("info",
        f"  outside unit circle: {int((mags >  1.0 + tol).sum())}/{len(eigvals)}")
    log("info", f"  spectral radius max|λ| = {spectral_radius:.6f}")

    eigenvalue_plot(
        eigvals,
        out_path=out_path,
        stable_tol=tol,
        title=f"{label} A-matrix eigenvalues on {C.DATASET.upper()}   "
              f"·   rank {rank}   ·   spectral radius = {spectral_radius:.4f}",
    )
    log("ok", f"saved {out_path}")


def do_plot_combined(methods) -> None:
    """Cross-method overlay: one figure with N+1 stacked panels (actual
    + each method), each panel showing the four classification metrics
    + the test loss over training steps. Reads the cached eval curves;
    requires plot_loss or plot_classification to have run for every
    method so the per-method cache file exists."""
    eid = E.get_or_init_exp()
    out_path = E.plot_path("overlay", "combined")
    banner(f"plot_combined start", exp=eid, methods=",".join(methods),
           output=out_path)

    snap = Recorder.load(E.snapshots_path())
    idx, grid_steps, fit_start_grid, plot_split = _grid(snap)
    snap_path = str(E.snapshots_path())

    actual = _load_curve_if_fresh(E.exp_dir() / "actual_eval_curve.npz",
                                  idx=idx, snap_path=snap_path)
    if actual is None:
        log("warn", "plot_combined: no actual_eval_curve cache; skipping")
        return

    panels = [("real network", actual)]
    for method in methods:
        info = get_method(method)
        cache_path = E.curves_dir() / f"{method}.npz"
        # we don't know the fingerprint without reloading the analysis -
        # but the per-method op chain (plot_loss / plot_classification)
        # always writes a fingerprint-consistent file before we get here.
        c = _load_curve_if_fresh(cache_path, idx=idx, snap_path=snap_path)
        if c is None:
            log("warn",
                f"plot_combined: skipping {method} (no fresh "
                f"curves cache; ensure plot_loss or plot_classification ran)")
            continue
        panels.append((f"{info['label']} forecast", c))

    if len(panels) < 2:
        log("warn", "plot_combined: nothing to overlay; skipping")
        return

    combined_overlay_plot(
        grid_steps, panels, plot_split,
        fit_start_idx=fit_start_grid,
        metric_keys=_CURVE_METRICS,
        out_path=out_path,
        title=f"All methods on {C.DATASET.upper()}   ·   "
              f"actual vs predicted networks   ·   "
              f"eval on {len(idx)} grid points",
    )
    log("ok", f"saved {out_path}")


# ---------------------------------------------------------------------------
# pipeline (used by run_exp.py)
# ---------------------------------------------------------------------------

_OPS_ORDER = ("analyze", "plot_eigenvalues", "plot_accuracy",
              "plot_loss", "plot_classification")
_OP_DISPATCH = {
    "analyze":             do_analyze,
    "plot_eigenvalues":    do_plot_eigenvalues,
    "plot_accuracy":       do_plot_accuracy,
    "plot_loss":           do_plot_loss,
    "plot_classification": do_plot_classification,
}

# Aggregator ops run once per pipeline (not per method) after the
# per-method loop. They depend on the curves caches produced by the
# per-method ops.
_AGG_OPS = ("plot_combined",)


def run_full_pipeline(methods: list[str] | tuple[str, ...] | None = None,
                      ops:     list[str] | tuple[str, ...] | None = None) -> None:
    """For each method in `methods` (or config.METHODS by default), run
    the operations listed in `ops` in canonical order:
    analyze -> plot_eigenvalues -> plot_accuracy -> plot_loss ->
    plot_classification. After the per-method loop, plot_combined is
    invoked once to write the cross-method overlay (skipped if no
    method-eval cache was produced this pipeline).

    `ops=None` runs everything. Pass a subset (e.g. ['analyze',
    'plot_classification']) to skip the others. plot_* ops require
    analyze to have been run already (in this experiment); the
    orchestrator picks the canonical order so `analyze` comes first
    when included.
    """
    if methods is None:
        methods = list(getattr(C, "METHODS", ()))
    if ops is None:
        ops = list(_OPS_ORDER) + list(_AGG_OPS)
    # canonicalise op order; aggregator ops are filtered separately
    per_method_ops = [o for o in _OPS_ORDER if o in ops]
    agg_ops        = [o for o in _AGG_OPS   if o in ops]
    unknown = [o for o in ops
               if o not in _OP_DISPATCH and o not in _AGG_OPS]
    if unknown:
        raise ValueError(f"unknown ops: {unknown}; "
                         f"valid: {list(_OP_DISPATCH) + list(_AGG_OPS)}")

    # Apply reproducibility tier (no-op for 'fast'; locks BLAS to one
    # thread + torch.set_num_threads(1) for 'analysis' or 'strict').
    from . import repro as R
    R.apply_for_analysis()

    _log_memory_estimate(methods)

    # Auto rank-selection for OptDMDc / cOptDMDc. Dispatches on
    # config.RANK_SELECTOR (see neural_dmd.rank_selectors). Skipped when
    # neither LM-based method appears in `methods` (saves the cost when
    # running only DMDc / sDMDc).
    if any(m in methods for m in ("optdmdc", "coptdmdc")) \
            and "analyze" in per_method_ops:
        try:
            from .rank_selectors import apply as _apply_rank_selector
            from .snapshots import Recorder
            from . import experiments as _E
            snap_path = _E.snapshots_path()
            if snap_path.exists():
                snap = Recorder.load(snap_path)
                selector_name = getattr(C, "RANK_SELECTOR", "fixed")
                log("info",
                    f"runners: applying rank selector '{selector_name}'")
                _apply_rank_selector(snap)
            else:
                log("warn",
                    f"runners: snapshots not found at {snap_path}; "
                    "skipping rank selector")
        except Exception as e:
            log("warn", f"runners: rank selector failed ({e}); falling back to "
                        "configured METHOD_PARAMS rank")

    try:
        for method in methods:
            banner(f"=== METHOD: {method}", ops=",".join(per_method_ops))
            for op in per_method_ops:
                _OP_DISPATCH[op](method)

        for op in agg_ops:
            if op == "plot_combined":
                do_plot_combined(methods)
    finally:
        # Evict the SVD cache so the persistent ~1-3 GB doesn't linger
        # past the end of the pipeline when the user re-uses the same
        # process for plotting / inspection.
        from .dmdc import _kernel_cache_clear, _SVD_CACHE
        if _SVD_CACHE:
            log("info",
                f"runners: clearing kernel SVD cache ({len(_SVD_CACHE)} entries) "
                f"to free memory")
            _kernel_cache_clear()


def _log_memory_estimate(methods) -> None:
    """Quick rough estimate of peak RAM the pipeline will consume + a
    psutil read of what's currently free. Helps users on smaller boxes
    spot the OOM risk before the SVD blows up."""
    try:
        from .snapshots import Recorder
        from . import experiments as E
        snap_path = E.snapshots_path()
        if not snap_path.exists():
            return
        snap = Recorder.load(snap_path)
        n, m = snap["X"].shape
        fit_split     = int(snap["fit_split"])
        fit_start_idx = int(snap.get("fit_start_idx", 0))
        m_fit = max(fit_split - fit_start_idx, 1)
        bytes_per = np.dtype(C.PRECISION).itemsize
        # Two SVDs (Omega + X_fit), each holds full Ux ~ n * m_fit + workspace
        svd_persistent = 2 * n * m_fit * bytes_per
        svd_peak_workspace = 2 * n * m_fit * bytes_per
        snap_in_mem = n * m * bytes_per
        rough_peak = (svd_persistent + svd_peak_workspace + snap_in_mem) / 1e9

        log("info",
            f"runners: rough peak RAM estimate at {C.PRECISION} = "
            f"{rough_peak:.1f} GB  (n={n} m_fit={m_fit} m_total={m})")

        try:
            import psutil
            vm = psutil.virtual_memory()
            log("info",
                f"runners: system memory  total={vm.total / 1e9:.1f} GB  "
                f"available={vm.available / 1e9:.1f} GB  used%={vm.percent:.0f}")
            if vm.available < rough_peak * 1.3 * 1e9:
                log("warn",
                    f"runners: estimated peak ({rough_peak:.1f} GB) is close "
                    f"to or exceeds available memory ({vm.available / 1e9:.1f} "
                    f"GB). Consider PRECISION='float32' (already set) or "
                    f"smaller EPOCHS / FIT_RANGE.")
        except ImportError:
            pass
    except Exception as e:
        log("warn", f"runners: memory estimate skipped ({e})")
