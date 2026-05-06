"""
Per-method analyze + plot dispatch. The thin script wrappers under
scripts/ all delegate here, and run_exp.py uses these directly to
orchestrate full pipelines.

Conventions:

    do_analyze(method)           reads snapshots, runs the method, saves
                                 analysis npz, calls log_eigenvalue_report,
                                 registers metrics in the experiment.
    do_plot_loss(method)
    do_plot_accuracy(method)     each reads its method's analysis npz and
    do_plot_eigenvalues(method)  writes a png under plots/<method>/<kind>.png

All file paths are resolved through neural_dmd.experiments. Per-method
hyperparameters come from neural_dmd.config.METHOD_PARAMS[method] and
are forwarded as kwargs to the registered run callable.
"""

from __future__ import annotations

import time

import numpy as np

from . import config as C
from . import experiments as E
from .figures import comparison_plot, eigenvalue_plot
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


def do_plot_loss(method: str) -> None:
    # Only import torch / data / metrics when actually plotting loss
    # (which needs forward passes through the network). Keeps eigvals /
    # accuracy plots fast even on machines without CUDA-ready torch.
    import torch  # noqa: F401
    from .data import get_loaders
    from .metrics import LOSSES, loss_at
    from .model import MLP

    info  = get_method(method)
    label = info["label"]

    eid = E.get_or_init_exp()
    out_path = E.plot_path(method, "loss")

    banner(f"plot_loss[{method}] start",
           exp=eid, dataset=C.DATASET, loss=C.LOSS, device=C.DEVICE,
           output=out_path)

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

    banner("evaluate loss curves", grid_points=len(idx),
           dataset=C.DATASET, loss=C.LOSS)
    _, test_loader = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)
    net = MLP(C.ARCH).to(C.DEVICE)
    loss_fn = LOSSES[C.LOSS]

    L_actual = _loss_curve("actual", net, snap["X"], idx, test_loader, loss_fn)
    L_pred   = _loss_curve(label,   net, X_pred,    idx, test_loader, loss_fn)

    summary = comparison_plot(
        grid_steps, L_actual, L_pred, plot_split,
        fit_start_idx=fit_start_grid,
        out_path=out_path,
        title=f"{label} loss-curve forecast on {C.DATASET.upper()}   "
              f"·   fit on {fit_split - cache.get('fit_start_idx', 0):d} dense snapshots   "
              f"·   eval on {len(idx)} grid points   ·   rank {rank}",
        ylabel=f"test {C.LOSS}",
        actual_label=r"actual   $\mathcal{L}(x_k;\mathcal{D})$",
        pred_label=fr"{label}   $\mathcal{{L}}(\hat{{x}}_k;\mathcal{{D}})$",
        summary_label="mean |Δloss|   in-sample = {in_:.4f}    out-of-sample = {out:.4f}",
    )
    log("ok", f"saved {out_path}")

    # Forecast-region final / minimum: tells you whether the prediction
    # ends up at the right loss value (final) and whether at any point in
    # the forecast it dipped to a value close to the actual minimum.
    metrics_payload = {
        "in_sample_dloss":   summary["in_sample"],
        "out_sample_dloss":  summary["out_sample"],
    }
    if plot_split < len(idx):
        L_act_fc = L_actual[plot_split:]
        L_prd_fc = L_pred[plot_split:]
        metrics_payload.update({
            "forecast_loss_actual_final": float(L_act_fc[-1]),
            "forecast_loss_pred_final":   float(L_prd_fc[-1]),
            "forecast_loss_actual_min":   float(L_act_fc.min()),
            "forecast_loss_pred_min":     float(L_prd_fc.min()),
        })
    E.write_metrics_bulk(method, metrics_payload)


def _loss_curve(label, net, X_cols, idx, test_loader, loss_fn):
    from .metrics import loss_at
    t = time.time()
    out = np.empty(len(idx), dtype=np.float64)
    for i, k in enumerate(idx, start=1):
        out[i - 1] = loss_at(net, X_cols[:, k], test_loader, C.DEVICE, loss_fn)
        progress(f"{label} loss curve", i, len(idx), t, every_pct=25.0)
    return out


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
        ylabel="prediction accuracy  [%]",
        actual_label=r"actual   $x_k$  (reference, 100%)",
        pred_label=fr"{label}   $100\cdot(1 - \|\hat{{x}}_k - x_k\|/\|x_k\|)$",
        summary_label="mean |Δaccuracy|   in-sample = {in_:.4f}%    out-of-sample = {out:.4f}%",
    )
    log("ok", f"saved {out_path}")
    E.write_metrics_bulk(method, {
        "in_sample_dacc_pct":  summary["in_sample"],
        "out_sample_dacc_pct": summary["out_sample"],
    })


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


# ---------------------------------------------------------------------------
# pipeline (used by run_exp.py)
# ---------------------------------------------------------------------------

_OPS_ORDER = ("analyze", "plot_eigenvalues", "plot_accuracy", "plot_loss")
_OP_DISPATCH = {
    "analyze":          do_analyze,
    "plot_eigenvalues": do_plot_eigenvalues,
    "plot_accuracy":    do_plot_accuracy,
    "plot_loss":        do_plot_loss,
}


def run_full_pipeline(methods: list[str] | tuple[str, ...] | None = None,
                      ops:     list[str] | tuple[str, ...] | None = None) -> None:
    """For each method in `methods` (or config.METHODS by default), run
    the operations listed in `ops` in canonical order:
    analyze -> plot_eigenvalues -> plot_accuracy -> plot_loss.

    `ops=None` runs all four. Pass a subset (e.g. ['analyze',
    'plot_accuracy']) to skip the others. plot_* ops require analyze to
    have been run already (in this experiment); the orchestrator picks
    the canonical order so `analyze` comes first when included.
    """
    if methods is None:
        methods = list(getattr(C, "METHODS", ()))
    if ops is None:
        ops = list(_OPS_ORDER)
    # canonicalise op order
    ops = [o for o in _OPS_ORDER if o in ops]
    unknown = [o for o in ops if o not in _OP_DISPATCH]
    if unknown:
        raise ValueError(f"unknown ops: {unknown}; valid: {list(_OP_DISPATCH)}")

    _log_memory_estimate(methods)

    try:
        for method in methods:
            banner(f"=== METHOD: {method}", ops=",".join(ops))
            for op in ops:
                _OP_DISPATCH[op](method)
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
