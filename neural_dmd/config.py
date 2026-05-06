"""
Single source of truth for every tunable knob.

Path resolution (where snapshots / model / analyses / plots actually
live on disk) is delegated to neural_dmd.experiments - this file only
declares values, never composes file paths under outputs/.

Knobs are grouped:

  EXPERIMENT          - identification + which methods to run
  TRAINING            - dataset, model architecture, optimiser, schedule
  SNAPSHOT RECORDER   - cadence + fit window
  METHOD PARAMS       - per-method hyperparameters (METHOD_PARAMS dict)
  NUMERICAL TOLERANCE - one shared knob for stability classification
  RUNTIME             - device etc.

To plug in something new:
  - dataset:       register a (Dataset class, transform) pair in data.DATASETS
  - loss / metric: register a callable in metrics.LOSSES or metrics.METRICS
  - architecture:  change ARCH (list of MLP layer widths)
  - control input: edit control_fn() below
  - method:        register in neural_dmd.methods.METHODS, add an entry
                   to METHOD_PARAMS, list it in METHODS
"""

from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# project root paths (raw filesystem only - everything else is dispatched
# through neural_dmd.experiments at runtime)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "data"            # input cache (MNIST, ...)
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"         # all generated artifacts live under here
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Backwards-compat alias used by some plot scripts that don't yet route
# through experiments.* (kept for transition; prefer the experiments
# module helpers in new code).
PLOTS_ROOT = OUTPUT_ROOT / "plots"


# ---------------------------------------------------------------------------
# EXPERIMENT
# ---------------------------------------------------------------------------

# Used to name the experiment directory:
#   outputs/experiments/<YYYY-MM-DD_HH-MM-SS>_<EXP_LABEL>/
EXP_LABEL       = "baseline"
EXP_DESCRIPTION = (
    "Fit on first 50% of the data, predict the next 50%.")

# Which methods to run for this experiment. Order is preserved in
# summary.md tables. Entries must exist in neural_dmd.methods.METHODS.
METHODS = ("dmdc", "sdmdc", "optdmdc", "coptdmdc")


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

DATASET    = "mnist"          # see data.DATASETS for registered options
BATCH_SIZE = 100              # train batch size (60000/100 = 600 steps/epoch exactly)
EVAL_BATCH = 1000             # eval batch size (no grads, can be larger)

# Layer widths for the MLP. The first must equal the flattened input
# dimension of the dataset (28*28=784 for MNIST, 32*32*3=3072 for CIFAR-10).
# The last must equal the number of classes.
ARCH = [784, 128, 64, 10]

EPOCHS = 10
LR     = 1e-3
LR_MIN = 0.0                  # cosine-annealing floor
LOSS   = "cross_entropy"      # name in metrics.LOSSES used for training + plot_loss
METRIC = "accuracy"           # name in metrics.METRICS used by test.py
SEED   = 0                    # torch + numpy seed; participates in data_hash


def control_fn(optimizer, step):
    """Return the control vector u_k recorded with each gradient step.

    Default: scalar learning rate. Replace with anything callable that
    returns a list / 1-D iterable of floats - DMDc will learn one column
    of B per control dimension.

    Called every step (not only on snapshot steps) so the recorder stores
    the full per-step control sequence U.
    """
    return [optimizer.param_groups[0]["lr"]]


# ---------------------------------------------------------------------------
# SNAPSHOT RECORDER + FIT WINDOW
#
# FIT_RANGE = (start, end)   dense snapshots in [start, end)
#                            sparse elsewhere  (= forecast cadence)
# FIT_RANGE = None           use FIT_FRAC: fit on [0, FIT_FRAC * total_steps)
#
# FIT_RANGE wins if both are set. SNAP_FIT_EVERY governs cadence inside
# the fit window; SNAP_FORECAST_EVERY governs cadence outside (both
# pre-fit and post-fit). Both are training-relevant: changing them
# invalidates the data_hash and forces a fresh training run.
# ---------------------------------------------------------------------------

SNAP_FIT_EVERY      = 1
SNAP_FORECAST_EVERY = 50

FIT_FRAC  = 0.5
FIT_RANGE = None

# Inject zero-mean Gaussian noise into every recorded parameter
# snapshot, simulating a noisy "measurement" of the underlying clean
# trajectory. NOISE_SIGMA is RELATIVE to the global trajectory std, so:
#
#   NOISE_SIGMA = 0.0    clean (default)
#   NOISE_SIGMA = 1e-3   ~paper-faithful (Rains et al. 2024 §3.1 noise study)
#   NOISE_SIGMA = 1e-2   heavy noise; lets cOpt's de-biasing shine vs DMDc
#
# Seeded deterministically from SEED so a given (training config,
# NOISE_SIGMA) pair always produces byte-identical noisy snapshots.
# Participates in the data_hash, so each noise level gets its own
# outputs/data/<hash>/ cache and you can sweep noise levels by changing
# this knob alone (no other config edits needed).
NOISE_SIGMA = 0.0

# ---------------------------------------------------------------------------
# METHOD PARAMS
#
# Per-method hyperparameters keyed by method name. Each method's run()
# is invoked with **METHOD_PARAMS[name].
#
# Memory note for OptDMDc / cOptDMDc: the dense Jacobian is
#     ~16 * (m_fit-1) * rank^2 bytes
# At rank=200 that's ~2 GB per LM iter. Keep rank modest unless you
# rebuild on top of a matrix-free LinearOperator path.
# ---------------------------------------------------------------------------

METHOD_PARAMS: dict[str, dict] = {
    "dmdc":     {"rank": None},
    "sdmdc":    {"rank": None,
                 "dt":   float(SNAP_FIT_EVERY)},
    "optdmdc":  {"rank":     50,
                 "pod_rank": None,
                 "max_iter": 50,
                 "tol":      1e-6,
                 # Early-stopping: quit when residual has improved by
                 # less than `tol_rel` for `patience` consecutive
                 # accepted LM iterations. Catches the regime where LM
                 # is no longer fitting signal and starting to overfit
                 # in-fit noise (which destroys forecast quality).
                 "tol_rel":  1e-3,
                 "patience": 3,
                 "gmax":     50,
                 "incr":     1.5,
                 "decr":     2.0,
                 "nu0":      2.0,
                 "dt":       float(SNAP_FIT_EVERY)},
    "coptdmdc": {"rank":     50,
                 "pod_rank": None,
                 "max_iter": 50,
                 "tol":      1e-6,
                 "tol_rel":  1e-3,
                 "patience": 3,
                 "gmax":     50,
                 "incr":     1.5,
                 "decr":     2.0,
                 "nu0":      2.0,
                 "dt":       float(SNAP_FIT_EVERY)},
}


# ---------------------------------------------------------------------------
# NUMERICAL TOLERANCE
# ---------------------------------------------------------------------------

# cOptDMDc puts radially-projected eigenvalues exactly on the unit
# circle (Re(gamma)=0 -> |exp(gamma*dt)| = sqrt(cos^2 + sin^2) = 1).
# In floating-point, np.abs of cos + i*sin may round to 1.0 +/- ~1e-16,
# so a strict "> 1" check classifies on-circle eigenvalues as unstable.
# Anything within tol of the unit circle is treated as stable.
EIG_STABLE_TOL = 1e-12


# ---------------------------------------------------------------------------
# PRECISION (memory budget)
#
# All large arrays (X, U, SVD outputs, Jacobian) are kept in this dtype.
# Small per-operator pieces (eigendecomposition of F, gamma vector,
# alpha/beta_star) stay in f64 so the eigenvalue / log / exp chain
# doesn't lose precision on the small p x p matrices.
#
#   "float32"  ~6-8 GB peak at the default 50-epoch baseline; runs on
#              16-32 GB machines with room to spare.  (default)
#   "float64"  ~13-15 GB peak; needs 32+ GB.  Use only when you've
#              observed numerical issues that f32 cannot resolve.
#
# Memory savings come from:
#   X snapshots cast in-place (half the size, no copy)
#   sgesdd vs dgesdd in numpy.linalg.svd (~half workspace + outputs)
#   smaller cached SVDs (Ux_full, Uo_full)
#   smaller Jacobian in OptDMDc / cOptDMDc (also halves rank^2 cost)
# ---------------------------------------------------------------------------

PRECISION = "float32"


# ---------------------------------------------------------------------------
# RUNTIME
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
