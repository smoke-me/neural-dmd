"""
Shared matplotlib helpers.

`comparison_plot`           two-curve figure (actual vs predicted) with
                            shaded fit region + fit/forecast split.
                            Used by plot_loss / plot_accuracy.
`eigenvalue_plot`           |λ| scatter on the unit circle.
`classification_grid_plot`  2x2 grid of accuracy / precision / recall /
                            F1, each panel showing the actual vs the
                            predicted network for one method.
`combined_overlay_plot`     One figure with N+1 stacked panels (actual
                            + each method); each panel overlays the
                            four classification metrics + loss over
                            training steps, with the fit/forecast split.

All four plot helpers share the same colour vocabulary
(_PALETTE_ACTUAL / _PALETTE_PRED / _METRIC_COLORS) so plots from
different ops read consistently in the summary.html report.
"""

import matplotlib.pyplot as plt
import numpy as np


# Palette shared across plots.
_C_ACTUAL = "#5ed3d3"   # cyan
_C_PRED   = "#ffb27a"   # warm orange
_C_SPLIT  = "#ffffff"   # fit/forecast separator

# Per-metric colours used by the classification grid + combined overlay.
# Keys match runners._CURVE_METRICS so callers can index by metric name.
_METRIC_COLORS = {
    "loss":      "#c084fc",   # violet
    "accuracy":  "#5ed3d3",   # cyan
    "precision": "#ffb27a",   # orange
    "recall":    "#fb7185",   # rose
    "f1":        "#86efac",   # mint
}

_METRIC_DISPLAY = {
    "loss":      "loss",
    "accuracy":  "accuracy",
    "precision": "precision",
    "recall":    "recall",
    "f1":        "F1",
}


def comparison_plot(steps, actual, predicted, *,
                    out_path, title, ylabel,
                    actual_label, pred_label, summary_label,
                    fit_start_step: int,
                    fit_end_step: int,
                    y_clip: tuple | None = None,
                    auto_clip_factor: float = 10.0):
    # steps           : (m,) x-axis (gradient-step index of each grid point)
    # actual          : (m,) y-values from the real x_k
    # predicted       : (m,) y-values from the DMDc forecast x_hat_k
    # fit_start_step  : gradient step where the fit window begins (drawn
    #                   as the left vertical guide when > min(steps))
    # fit_end_step    : gradient step where the fit window ends (= first
    #                   forecast step; drawn as the fit/forecast split).
    # title, ylabel   : figure decoration
    # *_label         : strings shown in legend / x-axis footer
    # y_clip          : explicit (y_lo, y_hi) bounds. None -> auto-clip
    #                   when predicted dwarfs actual.
    # auto_clip_factor: when |predicted|.max() exceeds this multiple of
    #                   |actual|.max(), or when predicted has any non-
    #                   finite values, clip y-axis to a window around
    #                   actual's range so the actual curve stays readable.
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)

    c_actual = _C_ACTUAL     # cyan - actual curve
    c_pred   = _C_PRED       # warm orange - DMDc curve
    c_fit_bg = _C_ACTUAL     # tint of the fit-region shaded band

    a = np.asarray(actual,    dtype=float)
    p = np.asarray(predicted, dtype=float)
    steps_arr = np.asarray(steps)
    m = len(steps_arr)
    # X positions for fit-window boundaries: use the gradient-step
    # numbers DIRECTLY (not steps[idx]), so the lines and the shaded
    # band land exactly on the configured fit window even when the eval
    # grid skips over the boundary value.
    fit_left_x  = float(fit_start_step)
    fit_right_x = float(fit_end_step)

    # Shaded background for the fit window only.
    ax.axvspan(fit_left_x, fit_right_x, color=c_fit_bg, alpha=0.06, lw=0)

    # Curves.
    ax.plot(steps, a, color=c_actual, lw=2.2, label=actual_label)
    ax.plot(steps, p, color=c_pred,   lw=2.0, ls="--", label=pred_label)

    # Right-side fit/forecast split (always drawn).
    ax.axvline(fit_right_x, color="#ffffff", ls="--", lw=1.6, alpha=0.7,
               label=f"fit / forecast split  (k={int(fit_right_x)})")
    # Left-side fit-window boundary (only when the fit window does not
    # start at step 0, i.e. when FIT_RANGE was set).
    if fit_start_step > (int(steps_arr[0]) if m > 0 else 0):
        ax.axvline(fit_left_x, color="#ffffff", ls="--", lw=1.2, alpha=0.5,
                   label=f"pre-fit boundary  (k={int(fit_left_x)})")

    # ----- y-axis clipping -----
    # If predicted explodes (NaN / inf / huge magnitude vs actual), clip
    # the y-axis to a window around actual's range so the actual curve
    # stays visible. Predicted points outside the window are drawn at
    # the boundary; their count is shown in an annotation so the reader
    # knows the curve is being cut off.
    p_finite_mask = np.isfinite(p)
    a_finite      = a[np.isfinite(a)]
    if a_finite.size == 0:
        y_lo, y_hi = (None, None)
    else:
        a_min, a_max = float(a_finite.min()), float(a_finite.max())
        a_span = max(a_max - a_min, 1e-12)

        if y_clip is not None:
            y_lo, y_hi = y_clip
        elif (not p_finite_mask.all()) or (
                p_finite_mask.any() and
                np.nanmax(np.abs(p[p_finite_mask])) >
                auto_clip_factor * max(abs(a_min), abs(a_max), a_span)):
            pad = 0.5 * a_span
            y_lo = a_min - pad
            y_hi = a_max + pad
        else:
            y_lo, y_hi = (None, None)

    if y_lo is not None and y_hi is not None:
        ax.set_ylim(y_lo, y_hi)
        n_above = int(((p > y_hi) | ~p_finite_mask).sum())
        n_below = int((p < y_lo).sum())
        n_off   = n_above + n_below
        if n_off > 0:
            p_show = p[p_finite_mask]
            extreme = (f"max|pred|={np.max(np.abs(p_show)):.2e}"
                       if p_show.size else "all NaN/inf")
            ax.text(
                0.99, 0.97,
                f"⚠ {n_off}/{m} predicted points off-scale ({extreme})",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color="#ff9999",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#000000aa",
                          edgecolor="#ff9999", linewidth=0.6))

    # split labels (placed AFTER ylim has been set so positioning is sane)
    y_top = ax.get_ylim()[1]
    ax.text(fit_right_x, y_top * 0.97, " ← fit",      color="#dddddd",
            fontsize=10, ha="right", va="top")
    ax.text(fit_right_x, y_top * 0.97, " forecast →", color="#dddddd",
            fontsize=10, ha="left",  va="top")

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.18)

    # Footer: mean absolute mismatch within / outside the fit window.
    # Use nanmean and skip non-finite points so a single overflowed
    # value doesn't poison the whole summary. Membership is keyed on
    # the gradient-step boundaries directly so the mask agrees with
    # the visually drawn shaded band.
    in_mask  = (steps_arr >= fit_start_step) & (steps_arr < fit_end_step)
    out_mask = ~in_mask
    def _safe_mean(mask):
        diff = np.abs(p[mask] - a[mask])
        finite = np.isfinite(diff)
        return float(diff[finite].mean()) if finite.any() else float("nan")
    in_  = _safe_mean(in_mask)
    out_ = _safe_mean(out_mask)
    ax.set_xlabel(
        "training step  $k$\n" + summary_label.format(in_=in_, out=out_),
        fontsize=9,
    )

    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return {"in_sample": in_, "out_sample": out_}


def eigenvalue_plot(eigenvalues, *, out_path, title, stable_tol: float = 0.0):
    # Scatter plot in the complex plane of the eigenvalues of the DMDc
    # operator A.
    #   - dashed circle = unit circle |lambda| = 1 (the stability boundary
    #     for a discrete-time linear system); drawn thin in a distinct
    #     magenta so it doesn't compete visually with the dots
    #   - dot color encodes whether each mode decays (cyan, |lambda| <= 1)
    #     or grows (warm orange, |lambda| > 1) when iterated forward
    #   - axis bounds adapt to whichever is larger: the unit circle or the
    #     furthest eigenvalue (so distant outliers are always visible)
    # `stable_tol` widens the "stable" classification window: a mode with
    # |lambda| <= 1 + stable_tol is shown stable. Used by cOptDMDc whose
    # radial projection puts eigenvalues exactly on the unit circle - in
    # floating-point those land at 1.0 +/- ~1e-16, and a strict cutoff
    # would mislabel them as unstable.
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)

    eig = np.asarray(eigenvalues).astype(np.complex128)
    mags = np.abs(eig)
    stable_mask = mags <= 1.0 + stable_tol
    n_stable   = int(stable_mask.sum())
    n_unstable = int((~stable_mask).sum())
    spectral_radius = float(mags.max()) if mags.size else 0.0

    c_circle   = "#ff4fa3"   # magenta - distinct from cyan and orange dots
    c_stable   = "#5ed3d3"   # cyan
    c_unstable = "#ffb27a"   # warm orange

    # Real and imaginary axes (background).
    ax.axhline(0, color="#444444", lw=0.6, zorder=0)
    ax.axvline(0, color="#444444", lw=0.6, zorder=0)

    # Unit circle (the stability boundary in discrete time). Drawn with
    # full alpha so it reads cleanly even when stable eigenvalues cluster
    # densely on top of it.
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    ax.plot(np.cos(theta), np.sin(theta), color=c_circle, ls=(0, (5, 3)),
            lw=1.0, alpha=1.0, zorder=1,
            label=r"unit circle  $|\lambda|=1$")

    # Eigenvalues sit on top of the circle (zorder=3). Stable dots are
    # smaller and more transparent so the magenta circle reads through
    # them when many eigenvalues land at |lambda|=1.
    ax.scatter(eig.real[stable_mask], eig.imag[stable_mask],
               s=14, c=c_stable, alpha=0.55, edgecolors="none", zorder=3,
               label=f"stable     $|\\lambda|\\leq 1$  ({n_stable})")
    ax.scatter(eig.real[~stable_mask], eig.imag[~stable_mask],
               s=26, c=c_unstable, alpha=0.95, edgecolors="none", zorder=4,
               label=f"unstable   $|\\lambda|>1$  ({n_unstable})")

    # Symmetric, square frame that always contains the unit circle AND
    # all eigenvalues with a 5% margin. Uses real/imag extents separately
    # so a single far-out eigenvalue still fits without distorting the
    # aspect ratio.
    extent = 1.0
    if eig.size:
        extent = max(extent, float(np.max(np.abs(eig.real))),
                              float(np.max(np.abs(eig.imag))))
    lim = 1.05 * extent + 0.10   # 5% margin + a small additive pad
    lim = max(lim, 1.15)         # always show a bit of room around |λ|=1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(alpha=0.18, zorder=0)
    ax.set_xlabel(
        r"Re($\lambda$)" + "\n"
        + f"spectral radius  max$|\\lambda|$ = {spectral_radius:.6f}",
        fontsize=9,
    )

    # Legend lives BELOW the axes so it can never overlap an eigenvalue
    # dot, and is laid out as a single horizontal row. Solid white
    # background + black text + black border so it reads cleanly off the
    # dark canvas.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#222222",
        framealpha=1.0,
        labelcolor="#000000",
        fontsize=9,
        borderpad=0.6,
        columnspacing=1.6,
        handletextpad=0.6,
    )

    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# classification-metric plots
# ---------------------------------------------------------------------------

def _decorate_split(ax, *, fit_start_step, fit_end_step, draw_left=True):
    """Common decoration: shaded fit band + fit/forecast split line(s).
    X positions are gradient-step numbers (matching the snapshot
    metadata), so the lines land exactly on the configured fit window
    no matter how the eval grid is sampled."""
    fit_left_x  = float(fit_start_step)
    fit_right_x = float(fit_end_step)
    ax.axvspan(fit_left_x, fit_right_x, color=_C_ACTUAL, alpha=0.06, lw=0)
    ax.axvline(fit_right_x, color=_C_SPLIT, ls="--", lw=1.4, alpha=0.65)
    if draw_left and fit_left_x > 0:
        ax.axvline(fit_left_x, color=_C_SPLIT, ls="--", lw=1.0, alpha=0.45)
    return fit_left_x, fit_right_x


def classification_grid_plot(steps, actual, pred, *,
                             fit_start_step: int,
                             fit_end_step:   int,
                             out_path, title,
                             actual_label="real network",
                             pred_label="forecast"):
    """2x2 panel grid of accuracy / precision / recall / F1.
    `actual` and `pred` are dicts keyed by metric name (matching
    runners._CURVE_METRICS). Each panel overlays the actual curve and
    the predicted curve with the same fit/forecast decoration as
    comparison_plot."""
    plt.style.use("dark_background")
    metrics_layout = [
        ("accuracy",  "test accuracy"),
        ("precision", "macro precision"),
        ("recall",    "macro recall"),
        ("f1",        "macro F1"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.2),
                              constrained_layout=True, sharex=True)

    for ax, (key, ylabel) in zip(axes.flat, metrics_layout):
        a = np.asarray(actual[key], dtype=float)
        p = np.asarray(pred[key],   dtype=float)
        _decorate_split(ax,
                        fit_start_step=fit_start_step,
                        fit_end_step=fit_end_step)
        ax.plot(steps, a, color=_C_ACTUAL, lw=2.0, label=actual_label)
        ax.plot(steps, p, color=_C_PRED,   lw=1.8, ls="--", label=pred_label)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.18)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"training step  $k$")

    # one legend for the whole grid (top-right panel)
    axes[0, 1].legend(frameon=False, loc="lower right")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def combined_overlay_plot(steps, panels, *,
                          fit_start_step: int,
                          fit_end_step:   int,
                          metric_keys,
                          out_path, title):
    """Stacked-panel figure: one row per (label, curve) entry in
    `panels` (typically: ("actual", actual_curve) followed by one entry
    per DMD method). Within each panel the four classification metrics
    + the test loss are drawn as separate lines over training steps.

    `metric_keys` is the canonical ordering of metric names (matching
    runners._CURVE_METRICS); curves are dicts whose keys include these.

    Loss is plotted on a secondary y-axis (right) so the [0, 1] cls
    metrics retain their natural scale on the primary axis - mixing
    them on one axis squashes the cls curves into a thin band when
    loss starts near 2 (e.g. uniform 10-way softmax).
    """
    plt.style.use("dark_background")
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 1, figsize=(13.5, 2.4 * n_rows + 0.6),
                              sharex=True, constrained_layout=True,
                              squeeze=False)
    axes = axes[:, 0]

    twin_axes = []
    for ax, (label, curve) in zip(axes, panels):
        _decorate_split(ax,
                        fit_start_step=fit_start_step,
                        fit_end_step=fit_end_step)
        for key in metric_keys:
            if key == "loss":
                continue
            y = np.asarray(curve[key], dtype=float)
            ax.plot(steps, y, color=_METRIC_COLORS[key], lw=1.6,
                    label=_METRIC_DISPLAY[key])
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel(f"{label}\n[cls metrics]", fontsize=9)
        ax.grid(alpha=0.18)

        ax2 = ax.twinx()
        L = np.asarray(curve["loss"], dtype=float)
        ax2.plot(steps, L, color=_METRIC_COLORS["loss"], lw=1.4, ls="--",
                 label=_METRIC_DISPLAY["loss"])
        ax2.set_ylabel("loss", fontsize=8, color=_METRIC_COLORS["loss"])
        ax2.tick_params(axis="y", labelcolor=_METRIC_COLORS["loss"])
        ax2.spines["right"].set_color(_METRIC_COLORS["loss"])
        twin_axes.append(ax2)

    # Shared legend across the top: cls handles from first panel's
    # primary axis + the loss handle from its twin.
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = twin_axes[0].get_legend_handles_labels()
    handles += h2
    labels  += l2
    axes[0].legend(handles, labels,
                   ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 1.28),
                   frameon=False, fontsize=9)

    axes[-1].set_xlabel(r"training step  $k$")
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
