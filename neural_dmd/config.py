"""
Single source of truth for everything tunable in the experiment.

Edit values in this file - downstream modules read from here, so you do
NOT need to touch other files for typical changes.

To plug in something completely new (a new dataset, a new loss, a new
control input):
  - dataset:        register a (Dataset class, transform) pair in data.DATASETS
  - loss / metric:  register a callable in metrics.LOSSES or metrics.METRICS
  - architecture:   change ARCH (list of layer sizes for the MLP)
  - control input:  edit control_fn() below to return whatever vector u_k
                    you want DMDc to learn against
"""

from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# project paths
#
# All paths are resolved relative to the project root (the parent of the
# `neural_dmd/` package directory) so the scripts work regardless of which
# directory you launch them from. Generated artifacts are organised into
# outputs/ and outputs/plots/ to keep the repo root clean.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"            # input cache (MNIST, ...)
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"         # generated artifacts
PLOTS_ROOT   = OUTPUT_ROOT  / "plots"           # generated figures
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

# --- dataset ---
DATASET    = "mnist"          # see data.DATASETS for the registered options
BATCH_SIZE = 100              # train batch size (60000/100 = 600 steps/epoch exactly)
EVAL_BATCH = 1000             # eval batch size (no grads, can be larger)

# --- model ---
# Layer widths for the MLP. The first must equal the flattened input
# dimension of the dataset (28*28=784 for MNIST, 32*32*3=3072 for CIFAR-10).
# The last must equal the number of classes.
ARCH = [784, 128, 64, 10]

# --- training ---
EPOCHS = 5
LR     = 1e-3
LR_MIN = 0.0                  # cosine-annealing floor
LOSS   = "cross_entropy"      # name in metrics.LOSSES used for training + plot_loss.py
METRIC = "accuracy"           # name in metrics.METRICS used by test.py

# --- snapshot recorder ---
# Two cadences. Inside the FIT region (the first FIT_FRAC of training) we
# record every SNAP_FIT_EVERY gradient steps - dense data so DMDc has a lot
# to fit on. Inside the FORECAST region we record every SNAP_FORECAST_EVERY
# gradient steps - sparse comparison points where we check how the
# forecast is doing.
SNAP_FIT_EVERY      = 1
SNAP_FORECAST_EVERY = 50


def control_fn(optimizer, step):
    """Return the control vector u_k recorded with each gradient step.

    Default: scalar learning rate. Replace with anything callable that
    returns a list / 1-D iterable of floats - DMDc will learn one column
    of B per control dimension.

    Called every step (not only on snapshot steps) so the recorder stores
    the full per-step control sequence U. The forecast iteration needs
    u_k for every step in the forecast region too.

    Examples:
        return [optimizer.param_groups[0]["lr"]]                       # lr only
        return [optimizer.param_groups[0]["lr"], some_other_signal]    # lr + extra
    """
    return [optimizer.param_groups[0]["lr"]]


# --- DMDc analysis ---
FIT_FRAC = 0.5                # fraction of training steps to FIT on; rest is forecast
RANK     = None               # truncation rank (None = full available)

# ---------------------------------------------------------------------------
# artifact paths (always under outputs/)
# ---------------------------------------------------------------------------
CKPT_PATH     = OUTPUT_ROOT / "model.pt"        # written by train.py
SNAP_PATH     = OUTPUT_ROOT / "snapshots.npz"   # written by train.py
ANALYSIS_PATH = OUTPUT_ROOT / "analysis.npz"    # written by analyze.py
LOSS_PLOT     = PLOTS_ROOT  / "dmdc_loss.png"
ACC_PLOT      = PLOTS_ROOT  / "dmdc_accuracy.png"
EIG_PLOT      = PLOTS_ROOT  / "dmdc_eigenvalues.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
