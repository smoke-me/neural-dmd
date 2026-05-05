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
                    actual_label, pred_label, summary_label):
    # steps         : (m,) x-axis (training step index of each snapshot)
    # actual        : (m,) y-values from the real x_k
    # predicted     : (m,) y-values from the DMDc forecast x_hat_k
    # split         : index where fit ends and forecast begins
    # title, ylabel : figure decoration
    # *_label       : strings shown in legend / x-axis footer
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)

    c_actual = "#5ed3d3"     # cyan - actual curve
    c_pred   = "#ffb27a"     # warm orange - DMDc curve
    c_fit_bg = "#5ed3d3"     # tint of the fit-region shaded band

    split_x = steps[split - 1]
    m = len(steps)

    # Shaded background for the fit region.
    ax.axvspan(steps[0], split_x, color=c_fit_bg, alpha=0.06, lw=0)

    # The two curves we are comparing.
    ax.plot(steps, actual,    color=c_actual, lw=2.2, label=actual_label)
    ax.plot(steps, predicted, color=c_pred,   lw=2.0, ls="--", label=pred_label)

    # Bold dashed marker for the boundary between fit and forecast.
    ax.axvline(split_x, color="#ffffff", ls="--", lw=1.6, alpha=0.7,
               label=f"fit / forecast split  (k={split_x})")
    y_top = ax.get_ylim()[1]
    ax.text(split_x, y_top * 0.97, " ← fit",      color="#dddddd",
            fontsize=10, ha="right", va="top")
    ax.text(split_x, y_top * 0.97, " forecast →", color="#dddddd",
            fontsize=10, ha="left",  va="top")

    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.18)

    # Footer: mean absolute curve mismatch in / out of sample.
    in_  = float(np.mean(np.abs(predicted[:split] - actual[:split])))
    out_ = float(np.mean(np.abs(predicted[split:] - actual[split:]))) if split < m else float("nan")
    ax.set_xlabel(
        "training step  $k$\n" + summary_label.format(in_=in_, out=out_),
        fontsize=9,
    )

    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def eigenvalue_plot(eigenvalues, *, out_path, title):
    # Scatter plot in the complex plane of the eigenvalues of the DMDc
    # operator A.
    #   - dashed circle = unit circle |lambda| = 1 (the stability boundary
    #     for a discrete-time linear system); drawn thin in a distinct
    #     magenta so it doesn't compete visually with the dots
    #   - dot color encodes whether each mode decays (cyan, |lambda| <= 1)
    #     or grows (warm orange, |lambda| > 1) when iterated forward
    #   - axis bounds adapt to whichever is larger: the unit circle or the
    #     furthest eigenvalue (so distant outliers are always visible)
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.0, 7.0), constrained_layout=True)

    eig = np.asarray(eigenvalues).astype(np.complex128)
    mags = np.abs(eig)
    stable_mask = mags <= 1.0
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
