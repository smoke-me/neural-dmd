"""
Shared matplotlib helper for the DMDc comparison plots.

`comparison_plot` draws a two-curve figure (actual vs predicted) with a
shaded fit region, a vertical fit/forecast split line, and a footer
summarising the in-sample / out-of-sample mismatch. Used by both plot.py
(loss) and eval.py (accuracy).
"""

import matplotlib.pyplot as plt
import numpy as np


def comparison_plot(steps, actual, predicted, split, *,
                    out_path, title, ylabel,
                    actual_label, pred_label, summary_label,
                    fit_start_idx: int = 0,
                    y_clip: tuple | None = None,
                    auto_clip_factor: float = 10.0):
    # steps         : (m,) x-axis (training step index of each snapshot)
    # actual        : (m,) y-values from the real x_k
    # predicted     : (m,) y-values from the DMDc forecast x_hat_k
    # split         : snapshot index where fit ends (one past the last
    #                 fit-region snapshot, i.e. fit_split / fit_end_idx)
    # fit_start_idx : snapshot index where fit begins; defaults to 0 for
    #                 the historical FIT_RANGE = None case
    # title, ylabel : figure decoration
    # *_label       : strings shown in legend / x-axis footer
    # y_clip        : explicit (y_lo, y_hi) bounds. None -> auto-clip when
    #                 predicted dwarfs actual (see auto_clip_factor).
    # auto_clip_factor: when |predicted|.max() exceeds this multiple of
    #                 |actual|.max(), or when predicted has any non-finite
    #                 values, clip y-axis to a window around actual's
    #                 range so the actual curve stays readable. Off-scale
    #                 predicted points are flagged in an inset.
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)

    c_actual = "#5ed3d3"     # cyan - actual curve
    c_pred   = "#ffb27a"     # warm orange - DMDc curve
    c_fit_bg = "#5ed3d3"     # tint of the fit-region shaded band

    a = np.asarray(actual,    dtype=float)
    p = np.asarray(predicted, dtype=float)
    m = len(steps)
    split = max(int(split), fit_start_idx + 1)
    fit_left_x  = steps[fit_start_idx]
    fit_right_x = steps[split - 1]

    # Shaded background for the fit window only.
    ax.axvspan(fit_left_x, fit_right_x, color=c_fit_bg, alpha=0.06, lw=0)

    # Curves.
    ax.plot(steps, a, color=c_actual, lw=2.2, label=actual_label)
    ax.plot(steps, p, color=c_pred,   lw=2.0, ls="--", label=pred_label)

    # Right-side fit/forecast split (always drawn).
    ax.axvline(fit_right_x, color="#ffffff", ls="--", lw=1.6, alpha=0.7,
               label=f"fit / forecast split  (k={fit_right_x})")
    # Left-side fit-window boundary (only when the fit window does not
    # start at step 0, i.e. when FIT_RANGE was set).
    if fit_start_idx > 0:
        ax.axvline(fit_left_x, color="#ffffff", ls="--", lw=1.2, alpha=0.5,
                   label=f"pre-fit boundary  (k={fit_left_x})")

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
    # value doesn't poison the whole summary.
    in_mask = np.zeros(m, dtype=bool)
    in_mask[fit_start_idx:split] = True
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
