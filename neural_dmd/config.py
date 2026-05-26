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
EXP_LABEL       = "Useful fit"
EXP_DESCRIPTION = (
    "Fit on 350 steps and predict the rest!")

# Which methods to run for this experiment. Order is preserved in
# summary.md tables. Entries must exist in neural_dmd.methods.METHODS.
METHODS = ("dmdc", "sdmdc", "optdmdc_direct", "coptdmdc_direct")


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

EPOCHS = 5
# Additional gradient steps to run *after* the full EPOCHS pass. Extends
# training rightward into the forecast region without changing the
# fit-window boundary. Total training length is EPOCHS*batches_per_epoch
# + EXTRA_STEPS. Participates in the data_hash (changing it retrains).
EXTRA_STEPS = 300
LR     = 1e-3
LR_MIN = 0.0                  # cosine-annealing floor
LOSS   = "cross_entropy"      # name in metrics.LOSSES used for training + plot_loss
METRIC = "accuracy"           # name in metrics.METRICS used by test.py
SEED   = 0                    # torch + numpy seed; participates in data_hash

# Optimizer selection. Name looked up in neural_dmd.optimizers.OPTIMIZERS.
# Per-optimizer kwargs live in OPTIMIZER_PARAMS[<name>]; missing entries
# fall back to the builder's defaults. Both knobs participate in the
# data_hash, so switching optimizer (or its params) invalidates the
# snapshot cache.
#
#   OPTIMIZER = "adam"   torch.optim.Adam (default)
#   OPTIMIZER = "sgd"    torch.optim.SGD  (plain SGD; set momentum > 0
#                                          via OPTIMIZER_PARAMS for SGDM)
OPTIMIZER = "adam"

OPTIMIZER_PARAMS: dict[str, dict] = {
    "adam": {"betas":        (0.9, 0.999),
             "eps":          1e-8,
             "weight_decay": 0.0},
    "sgd":  {"momentum":     0.0,
             "dampening":    0.0,
             "weight_decay": 0.0,
             "nesterov":     False},
}


# Per-step control vector u_k for DMDc (driving x_{k+1} = A x_k + B u_k).
# The control is composed by concatenating the outputs of every source
# named in CONTROLS, in order, into one (q,) vector. See
# neural_dmd.controls for the registry of available sources and how to
# add new ones.
#
#   "lr"         scalar learning rate (1 dim). Reproduces the legacy
#                pre-controls behaviour exactly when used alone.
#   "batch_pca"  k-dim projection of the current batch's mean image onto
#                the top-k principal directions of the training set
#                (computed once at startup). Exposes "what kind of data
#                drove this step" to DMDc so its B matrix can map batch
#                content to weight-update direction.
#
# Per-source kwargs go in CONTROL_PARAMS[name]. Only the entries for
# active sources (those listed in CONTROLS) participate in the
# data_hash, mirroring the OPTIMIZER_PARAMS pattern: editing kwargs for
# an inactive source does NOT invalidate the snapshot cache.
CONTROLS = ("lr", "batch_pca")

CONTROL_PARAMS: dict[str, dict] = {
    "lr":        {},
    "batch_pca": {"k": 8},
}


# ---------------------------------------------------------------------------
# SNAPSHOT RECORDER + FIT WINDOW
#
# Pick ONE of the two knobs below; the other MUST be None. Setting both
# (or neither) raises a config error at startup - the previous
# "FIT_RANGE silently wins" behaviour was easy to misread.
#
#   FIT_FRAC                  fraction-based fit window
#     = 0.5                   -> fit on [0, 50%) of training
#     = (0.2, 0.8)            -> fit on [20%, 80%) of training
#
#   FIT_RANGE                 step-based fit window
#     = (start_step, end_step) -> fit on [start_step, end_step)
#                                must satisfy 0 <= start < end <= total_steps
#
# SNAP_FIT_EVERY governs the dense snapshot cadence INSIDE the fit
# window; SNAP_FORECAST_EVERY governs the sparse cadence outside (both
# pre-fit and post-fit regions). Every knob in this block participates
# in the data_hash: changing any value invalidates the snapshot cache
# and forces a fresh training run.
# ---------------------------------------------------------------------------

SNAP_FIT_EVERY      = 1
SNAP_FORECAST_EVERY = 50

# Exactly one of these two MUST be non-None; the other MUST be None.
FIT_FRAC  = None        # fraction-based: float OR (start_frac, end_frac)
FIT_RANGE = (150, 500)    # step-based:    (start_step, end_step) or None

# Inject zero-mean Gaussian noise into every recorded parameter
# snapshot, simulating a noisy "measurement" of the underlying clean
# trajectory. NOISE_SIGMA is RELATIVE to the global trajectory std, so:
#
#   NOISE_SIGMA = 0.0    OFF: clean snapshots, no noise added (default)
#   NOISE_SIGMA = 1e-3   ~paper-faithful (Rains et al. 2024 §3.1 noise study)
#   NOISE_SIGMA = 1e-2   heavy noise; lets cOpt's de-biasing shine vs DMDc
#
# Seeded deterministically from SEED so a given (training config,
# NOISE_SIGMA) pair always produces byte-identical noisy snapshots.
# Participates in the data_hash, so each noise level gets its own
# outputs/data/<hash>/ cache and you can sweep noise levels by changing
# this knob alone (no other config edits needed).
#
# To turn noise OFF: set NOISE_SIGMA = 0.0 (or any value <= 0). The
# inject_snapshot_noise call becomes a no-op.
NOISE_SIGMA = 0.0

# Per-checkpoint normalization for the DMD analysis. Name in
# neural_dmd.normalizers.NORMALIZERS. Removes whole-trajectory drift in
# weight MAGNITUDE so DMD models only the shape of the parameter tensors
# over time, not their overall size.
#
# IMPORTANT: snapshots on disk stay UN-normalized regardless of this
# knob - it is applied as an analysis-time pre/post-processing step
# around each DMD method (see neural_dmd.runners.do_analyze):
#   * X_fit normalized per-tensor before the DMD fit (operator learns shape).
#   * X_pred denormalized per-tensor on the way out (eval / plot pipelines
#     see genuine, original-scale weights, so real and forecasted
#     network test-loss curves are physically meaningful).
# Flipping this knob does NOT invalidate the snapshot data_hash; you can
# re-analyse the same cached snapshots under different normalizers.
#
#   "off"         No rescaling. DMD fits original-scale weights directly.
#   "per_tensor"  Each parameter tensor divided by its own L2 norm column
#                 by column. DMD operates on a unit-norm copy; predictions
#                 are scaled back using the recorded per-snapshot scales
#                 (forecast-region scales come from the recorded out-of-fit
#                 snapshots, so the comparison isolates shape error from
#                 magnitude drift).
SNAPSHOT_NORM = "per_tensor"

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
                 #
                 # To turn early stopping OFF: set tol_rel = 0.0 (or any
                 # value <= 0) OR patience = 0. The patience criterion
                 # then becomes inert; only the absolute `tol` and the
                 # outer `max_iter` cap can stop the loop.
                 "tol_rel":  0,
                 "patience": 0,
                 "gmax":     50,
                 "incr":     1.5,
                 "decr":     2.0,
                 "nu0":      2.0,
                 "dt":       float(SNAP_FIT_EVERY)},
    "coptdmdc": {"rank":     50,
                 "pod_rank": None,
                 "max_iter": 50,
                 "tol":      1e-6,
                 # tol_rel = 0.0 disables early stopping (see optdmdc above)
                 "tol_rel":  0,
                 "patience": 0,
                 "gmax":     50,
                 "incr":     1.5,
                 "decr":     2.0,
                 "nu0":      2.0,
                 "dt":       float(SNAP_FIT_EVERY)},
    # Direct-λ variants take the same kwargs (dt is accepted for API
    # symmetry but unused; λ already absorbs dt).
    "optdmdc_direct":  {"rank":     50,
                        "pod_rank": None,
                        "max_iter": 50,
                        "tol":      1e-6,
                        "tol_rel":  0,
                        "patience": 0,
                        "gmax":     50,
                        "incr":     1.5,
                        "decr":     2.0,
                        "nu0":      2.0,
                        "dt":       float(SNAP_FIT_EVERY)},
    "coptdmdc_direct": {"rank":     50,
                        "pod_rank": None,
                        "max_iter": 50,
                        "tol":      1e-6,
                        "tol_rel":  0,
                        "patience": 0,
                        "gmax":     50,
                        "incr":     1.5,
                        "decr":     2.0,
                        "nu0":      2.0,
                        "dt":       float(SNAP_FIT_EVERY)},
}


# ---------------------------------------------------------------------------
# RANK SELECTOR (for OptDMDc / cOptDMDc)
#
# Strategy name looked up in neural_dmd.rank_selectors.RANK_SELECTORS.
# The selected strategy runs once before the per-method analyses and
# patches METHOD_PARAMS["optdmdc"|"coptdmdc"]["rank"] in place so both
# LM methods agree on the same rank.
#
#   "fixed"          No-op. Keep METHOD_PARAMS ranks as user-configured.
#                    Use when you want to drive each method by hand.
#
#   "scan"           Held-out forecast-error scan over LM_RANK_AUTO_CANDIDATES
#                    (the historical LM_RANK_AUTO path). Picks the rank
#                    with lowest val L2 forecast error. Cost: 1-3 min at
#                    our default scale (uses the kernel's SVD cache).
#
#   "gavish_donoho"  Optimal hard SVD threshold (Gavish & Donoho 2014).
#                    Pure statistical pick from the X_fit spectrum - no
#                    forecast loop. Orders of magnitude cheaper than
#                    "scan". Honors GD_USE_KNOWN_SIGMA below.
#
# LM_RANK_AUTO is the legacy boolean; ignored when RANK_SELECTOR is set
# to anything other than "fixed". Kept here so the "scan" selector reads
# the same candidate list / patience / val_frac as before.
# ---------------------------------------------------------------------------

RANK_SELECTOR = "gavish_donoho"

# Gavish-Donoho options (consulted only when RANK_SELECTOR = "gavish_donoho").
#   GD_USE_KNOWN_SIGMA = False  -> median-based estimator (recommended;
#                                  needs no extra information beyond the
#                                  observed singular spectrum).
#   GD_USE_KNOWN_SIGMA = True   -> use NOISE_SIGMA above as the absolute
#                                  noise level (scaled by trajectory std,
#                                  matching how the recorder injected it).
GD_USE_KNOWN_SIGMA = False

LM_RANK_AUTO            = True   # legacy flag, see note above
LM_RANK_AUTO_CANDIDATES = (1, 5, 10, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 50, 100, 200, 400)
LM_RANK_AUTO_PATIENCE   = 2          # consecutive non-improving ranks before stop
LM_RANK_AUTO_VAL_FRAC   = 0.1        # last 10% of fit window held out for validation
LM_RANK_AUTO_MIN_IMPROV = 1e-3       # require >0.1% relative improvement to count


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


# ---------------------------------------------------------------------------
# REPRODUCIBILITY TIER
#
# Controls bit-exactness vs speed trade-off across machines.
#
#   "fast"     (default) Multi-threaded BLAS, GPU training when available.
#              Same-machine runs are bit-exact (full PyTorch determinism
#              stack is set in train.py). Cross-machine: differs at
#              4-5 sigfigs because BLAS implementations / instruction
#              sets / thread counts vary across hosts.
#
#   "analysis" Multi-threaded training (fast); single-threaded BLAS +
#              torch threads at the analysis stage. Train cost
#              unaffected. Analysis stage runs ~3-5x slower but is
#              bit-exact across same-architecture machines (x86_64 /
#              ARM separately) running matched library versions, when
#              fed identical snapshot bytes (combine with sharing
#              snapshots.npz across machines for full pipeline
#              reproducibility).
#
#   "strict"  Forces CPU + single-threaded everywhere from training
#              onwards. ~5-8x slower full pipeline. Bit-exact across
#              same-architecture machines with matched libraries.
#
# Cross-architecture bit-exactness (e.g. x86_64 vs Apple Silicon) is
# NOT achievable without identical hardware / Docker containers; see
# the "Reproducibility" section in README.md.
# ---------------------------------------------------------------------------

REPRO_TIER = "fast"
