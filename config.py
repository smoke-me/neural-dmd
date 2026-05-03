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

import torch

# --- dataset ---
DATASET    = "mnist"          # see data.DATASETS for the registered options
BATCH_SIZE = 64               # train batch size
EVAL_BATCH = 1000             # eval batch size (no grads, can be larger)
DATA_ROOT  = "./data"         # where torchvision caches the raw files

# --- model ---
# Layer widths for the MLP. The first must equal the flattened input
# dimension of the dataset (28*28=784 for MNIST, 32*32*3=3072 for CIFAR-10).
# The last must equal the number of classes.
ARCH = [784, 128, 64, 10]

# --- training ---
EPOCHS = 5
LR     = 1e-3
LOSS   = "cross_entropy"      # name in metrics.LOSSES used for training + plot.py
METRIC = "accuracy"           # name in metrics.METRICS used by eval.py

# --- snapshot recorder ---
# Capture (x_k, u_k) every N gradient steps. Fewer = less disk + faster
# DMDc fit but coarser temporal resolution.
SNAP_EVERY = 50


def control_fn(optimizer, step):
    """Return the control vector u_k recorded with each snapshot.

    Default: scalar learning rate. Replace with anything callable that
    returns a list / 1-D iterable of floats - DMDc will learn one column
    of B per control dimension.

    Examples:
        return [optimizer.param_groups[0]["lr"]]                       # lr only
        return [optimizer.param_groups[0]["lr"], some_other_signal]    # lr + extra
    """
    return [optimizer.param_groups[0]["lr"]]


# --- DMDc analysis ---
FIT_FRAC = 0.5                # fraction of trajectory to FIT on; rest is forecast
RANK     = None               # truncation rank (None = full available, ~n_fit-1)

# --- paths + device ---
CKPT_PATH = "model.pt"
SNAP_PATH = "snapshots.npz"
LOSS_PLOT = "dmdc_loss.png"
ACC_PLOT  = "dmdc_accuracy.png"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
