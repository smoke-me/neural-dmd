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
