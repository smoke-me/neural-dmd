# neural-dmd

Train an MLP on a classification dataset, record its parameter
trajectory during training, fit a family of DMDc-style linear models to
that trajectory, and measure how well each model forecasts the trained
network's behaviour over portions of the training run not used for
fitting.

Four methods are implemented:

| Method     | Idea                                                         |
|------------|--------------------------------------------------------------|
| `dmdc`     | Plain DMDc (Proctor-Brunton-Kutz, 2016) - SVD-based linear LSQ. |
| `sdmdc`    | DMDc + post-hoc radial-projection clipping of unstable eigenvalues. No optimisation. |
| `optdmdc`  | Optimized DMD with control (Askham-Kutz variable projection extended with exogeneous inputs). LM, no constraint. |
| `coptdmdc` | Constrained Optimized DMDc (Rains et al., JCP 2024). LM with `Re(γ) ≤ 0` enforced. |

Each method shares the same warm-start kernel and the same forecast
loop, so the differences between them are isolated to a small number
of well-marked stages.

---

## Quick start

```bash
# 1. install dependencies
uv venv
uv pip install -r requirements.txt

# 2. run the full pipeline (train if needed + every method in
#    config.METHODS + plots + summary.html)
python scripts/run_exp.py
```

Open `outputs/experiments/<exp_id>/summary.html` for the result. The
summary embeds the metrics table, eigenvalue diagnostics, and links
to every plot.

---

## Experiment system

The whole project is organised around a two-tier cache:

```
outputs/
  data/<data_hash>/                    training artifacts (content-addressed)
    snapshots.npz
    model.pt
    train_config.json
    train.log
  experiments/<exp_id>/                analysis runs (one per invocation)
    manifest.json                      exp metadata
    config.json                        full config snapshot
    metrics.json                       per-method metrics (incremental)
    run.log                            tee'd stdout/stderr
    analyses/<method>.npz              one per method that ran
    plots/<method>/{loss, accuracy, eigenvalues}.png
    summary.html
```

### Data cache (`outputs/data/<data_hash>/`)

`<data_hash>` is the first 12 hex digits of `sha256(<training-relevant
config>)`. The exact fields hashed live in
`neural_dmd.experiments.TRAIN_FIELDS`:

```
DATASET, ARCH, EPOCHS, BATCH_SIZE, LR, LR_MIN, LOSS,
FIT_FRAC, FIT_RANGE,
SNAP_FIT_EVERY, SNAP_FORECAST_EVERY,
SEED, NOISE_SIGMA,
+ source code of control_fn
```

Change any of those → `data_hash` flips → fresh training run. Change
anything else (`RANK`, `METHOD_PARAMS`, `EXP_LABEL`, ...) → cache hit,
training is reused, only the analyses re-run.

`scripts/train.py` is also content-addressed: it skips entirely if the
target `outputs/data/<hash>/` already has both `snapshots.npz` and
`model.pt`. Use `--force` to retrain anyway.

### Experiments (`outputs/experiments/<exp_id>/`)

`<exp_id>` = `<YYYY-MM-DD_HH-MM-SS>_<EXP_LABEL>`. Always sortable;
unique within a wall-clock second.

Set `EXP_LABEL` (and optionally `EXP_DESCRIPTION`) in `config.py` to
name the run. Different labels live side-by-side under
`outputs/experiments/`; nothing is overwritten across runs.

The orchestrator (`scripts/run_exp.py`) is the canonical entry point.
It

1. Mints a fresh `exp_id` and creates the directory structure.
2. Tees `stdout`/`stderr` into `run.log`.
3. Computes `data_hash`. Cache miss → run `train.py`. Cache hit → log
   "data cache HIT, reusing".
4. For each method in `config.METHODS`, runs analyse + 3 plots.
5. Writes `summary.html` from the accumulated `metrics.json`.

### Active experiment (env var)

`NEURAL_DMD_EXP_ID` selects the active experiment for any script
invocation. `run_exp.py` sets it implicitly so child analyses inherit
it. Standalone scripts (e.g. `scripts/plot_loss_dmdc.py`) resolve the
active experiment in this order:

1. `NEURAL_DMD_EXP_ID` env var (use it if set).
2. Lexicographically-latest existing experiment under
   `outputs/experiments/`.
3. A freshly-minted experiment.

So to re-run a single plot inside an existing experiment:

```bash
# implicit (uses the latest experiment)
python scripts/plot_eigenvalues_dmdc.py

# explicit
NEURAL_DMD_EXP_ID=2026-05-05_04-08-09_plateau-test \
  python scripts/plot_eigenvalues_dmdc.py
```

---

## Configuration

Everything lives in `neural_dmd/config.py`. Knobs are grouped into:

| Group                 | Examples                                                    |
|-----------------------|-------------------------------------------------------------|
| `EXPERIMENT`          | `EXP_LABEL`, `EXP_DESCRIPTION`, `METHODS`                   |
| `TRAINING`            | `DATASET`, `ARCH`, `EPOCHS`, `LR`, `LOSS`, `SEED`           |
| `SNAPSHOT RECORDER`   | `SNAP_FIT_EVERY`, `SNAP_FORECAST_EVERY`, `FIT_FRAC`, `FIT_RANGE` |
| `METHOD PARAMS`       | `METHOD_PARAMS["<method>"]` dict (rank, LM hyperparams, dt) |
| `NUMERICAL TOLERANCE` | `EIG_STABLE_TOL`                                            |
| `RUNTIME`             | `DEVICE`                                                    |

### Picking which methods to run

```python
METHODS = ("dmdc", "sdmdc", "optdmdc", "coptdmdc")   # everything (default)
METHODS = ("dmdc",)                                  # baseline only
METHODS = ("dmdc", "sdmdc")                          # cheap variants
METHODS = ("optdmdc", "coptdmdc")                    # LM-based variants
```

`run_exp.py --methods a,b,...` overrides for a single invocation
without editing the config.

### Per-method hyperparameters

```python
METHOD_PARAMS = {
    "dmdc":     {"rank": None},
    "sdmdc":    {"rank": None, "dt": float(SNAP_FIT_EVERY)},
    "optdmdc":  {"rank": 50, "pod_rank": None,
                 "max_iter": 50, "tol": 1e-6,
                 "tol_rel": 1e-3, "patience": 3,
                 "gmax": 50,
                 "incr": 1.5, "decr": 2.0, "nu0": 2.0,
                 "dt":  float(SNAP_FIT_EVERY)},
    "coptdmdc": {"rank": 50, "pod_rank": None,
                 "max_iter": 50, "tol": 1e-6,
                 "tol_rel": 1e-3, "patience": 3,
                 "gmax": 50,
                 "incr": 1.5, "decr": 2.0, "nu0": 2.0,
                 "dt":  float(SNAP_FIT_EVERY)},
}
```

`rank` is the LM operator size (memory-bounded, since the dense Jacobian
scales like `~16 * (m_fit-1) * rank^2` bytes). `pod_rank` is decoupled
and only governs the in-sample reconstruction lift, so the in-fit
accuracy plot can show ~100% even when the dynamics live in a small
subspace.

### Auto rank scan (Opt / cOpt)

```python
LM_RANK_AUTO            = False                       # default: OFF
LM_RANK_AUTO_CANDIDATES = (10, 25, 50, 100, 200, 400)
LM_RANK_AUTO_PATIENCE   = 2
LM_RANK_AUTO_VAL_FRAC   = 0.1
LM_RANK_AUTO_MIN_IMPROV = 1e-3
```

When `LM_RANK_AUTO = True`, the pipeline runs a held-out forecast scan
**before** OptDMDc / cOptDMDc analyze, picks the rank with lowest
validation forecast error, and patches both methods' `rank` in
`METHOD_PARAMS` so they agree.

**Procedure:**
1. Hold out the last `LM_RANK_AUTO_VAL_FRAC` (default 10%) of the fit window.
2. For each rank in `LM_RANK_AUTO_CANDIDATES` (ascending):
   - Fit the DMDc warm start on the remaining 90%.
   - Forecast across the held-out segment.
   - Compute mean relative L2 error.
3. Track best rank. Stop early when `LM_RANK_AUTO_PATIENCE` consecutive
   ranks fail to improve on the running best by `LM_RANK_AUTO_MIN_IMPROV`
   (relative).

**Why this is cheap:** the SVD cache in `dmdc._kernel` ensures the full
O(n·m²) factorisation runs only **once** per scan; every candidate
rank after that just slices the cached SVDs and rebuilds the small
reduced operator. Total scan cost ≈ 1–3 minutes at default scale,
regardless of how many candidates are listed.

**Why warm-start (no LM)?** The full LM loop would make the scan
prohibitively slow. The DMDc warm start is what Opt/cOpt initialise
from anyway, and forecast quality of the warm start tracks the
forecast quality of the optimised result closely enough to pick a
good rank.

**To turn OFF:** `LM_RANK_AUTO = False` (default). The static
`METHOD_PARAMS["optdmdc"]["rank"]` / `["coptdmdc"]["rank"]` are used.

The chosen rank + per-candidate validation error are saved under
`metrics.json["__rank_scan__"]` for the experiment.

### LM early stopping (Opt / cOpt)

Both `optdmdc.run` and `coptdmdc.run` halt the Levenberg-Marquardt
iteration as soon as one of these fires:

| Criterion | Condition | Knob |
|-----------|-----------|------|
| Absolute  | residual < `tol`                             | `tol`      |
| Patience  | relative improvement `(best − now)/best < tol_rel` for `patience` consecutive accepted iters | `tol_rel`, `patience` |
| Hard cap  | iteration count = `max_iter`                 | `max_iter` |
| Divergence | LM restart count > `gmax`                   | `gmax`     |

**Why patience?** At low LM rank (e.g. 50) on a richer trajectory, LM
keeps lowering the in-fit residual past the point of fitting signal;
those late iterations chase in-window noise that doesn't generalise to
the forecast region. The patience criterion stops once per-iter
improvement falls below `tol_rel`, capturing the model just before it
starts overfitting. The `lm_stop_reason` field in `metrics.json` (and
the `LM stop` column in `summary.html`) records which criterion fired.

Defaults `tol_rel = 1e-3, patience = 3` mean: "stop when the residual
hasn't dropped by 0.1% per iter for 3 iters in a row". Tighter
(`tol_rel = 1e-4, patience = 5`) lets LM run longer; looser
(`tol_rel = 5e-3, patience = 1`) stops sooner.

**To turn early stopping off**: set `tol_rel = 0.0` (or `patience = 0`)
in `METHOD_PARAMS["optdmdc"]` / `METHOD_PARAMS["coptdmdc"]`. The
patience criterion goes inert; only the absolute `tol` and the outer
`max_iter` cap can stop the loop. Use this when you want every
iteration spent — useful for diagnostic comparisons or when you
genuinely want LM to chase the residual all the way down.

### Snapshot noise (`NOISE_SIGMA`)

```python
NOISE_SIGMA = 0.0       # OFF: clean snapshots, no noise added (default)
NOISE_SIGMA = 1e-3      # paper-faithful (Rains et al. 2024 §3.1)
NOISE_SIGMA = 1e-2      # heavy noise; cOpt's de-biasing should beat DMDc here
```

**To turn noise off**: set `NOISE_SIGMA = 0.0` (or any value `<= 0`).
The injection call becomes a no-op; recorded snapshots are byte-clean.

When `NOISE_SIGMA > 0`, every recorded parameter snapshot has zero-mean
Gaussian noise added before `snapshots.npz` is written. The standard
deviation is interpreted **relative to the global trajectory std**:

```
sigma_abs = NOISE_SIGMA * std(X_full)
```

so the level scales sensibly across architectures / convergence stages.
Seeded deterministically from `SEED` so the same `(training config,
NOISE_SIGMA)` pair always produces byte-identical noisy snapshots.

`NOISE_SIGMA` is in `TRAIN_FIELDS`, so each value gets its own
`outputs/data/<hash>/` cache. To sweep noise levels, edit just this
knob and run `scripts/run_exp.py` — it'll retrain and produce a new
experiment dir per noise level, side-by-side.

This is the canonical knob to enable when you want to test whether
**OptDMDc / cOptDMDc actually beat DMDc**. Plain DMDc fits noise as
spurious eigenvalues; OptDMDc's variable-projection LSQ is provably
less biased on noisy data; cOpt additionally constrains the spurious
modes inside the unit circle. The differences are dramatic when noise
is large enough to corrupt DMDc but invisible when data is clean.

### Fit-window selection (`FIT_RANGE`)

```python
FIT_FRAC  = 0.5          # fit on [0, FIT_FRAC * total_steps)
FIT_RANGE = None         # falls back to FIT_FRAC

FIT_RANGE = (3000, 5000) # explicit fit window in gradient steps
```

`FIT_RANGE` is participating in `data_hash`, so changing it forces a
fresh training run. Use case: fit on a converged plateau region by
setting `FIT_RANGE = (start_of_plateau, end_of_plateau)` and bumping
`EPOCHS` so total_steps extends past `end_of_plateau`.

---

## Adding a new method

1. Implement `neural_dmd/<name>.py` with a `run(snap, **kwargs)`
   returning a dict containing at least
   `{X_pred, A, eigenvalues (or A only), rank, fit_split}`.
2. Add it to `neural_dmd.methods.METHODS` (label, run callable, has_lm
   flag, blurb).
3. Add hyperparameters in `config.METHOD_PARAMS["<name>"]` (or empty
   dict if none).
4. Mention the method in `config.METHODS` so it runs in the
   experiment.

The orchestrator + thin per-method scripts (`scripts/analyze_<name>.py`
etc.) pick it up automatically. Generate the four wrappers with the
heredoc loop in this README's `Add a method` cookbook (see
`scripts/run_exp.py` source for the pattern).

---

## What `summary.html` looks like

A self-contained HTML page (open in any browser; PNG plots are inlined
as base64 data URLs so the file is portable). Sections in order:

1. **Header** — exp_id, label, description (italic call-out), run time,
   data hash, list of methods that ran.
2. **Metrics** table — one row per method, columns:
   `Rank | POD rank | Spectral ρ | λ>1 | In Δacc% | Out Δacc% | In Δloss
   | Out Δloss | LM iters | LM stop | Wall (s)`.
   Cells with `λ>1 > 0` or NaN are highlighted red; `λ>1 = 0` green.
   `LM iters` + `LM stop` show how many Levenberg-Marquardt iterations
   actually ran (vs `max_iter`) and which stop criterion fired
   (`abs_tol`, `patience`, `max_iter`, `gmax exceeded`).
3. **Forecast-region final / min** table — `actual final L | pred final
   L | actual min L | pred min L | actual final acc% | pred final acc%`
   (only when loss plots ran).
4. **Eigenvalue stability** — for each method, either a green "all
   eigenvalues inside unit circle" note, or an `<h3>` + monospaced
   `λ_N = re+imj  |λ|=mag  excess=±N` list of every eigenvalue outside
   `1 + EIG_STABLE_TOL`.
5. **LM convergence** (only for OptDMDc / cOptDMDc when present) -
   initial → final residual + iter count + restart count.
6. **Plots** — one card per method, three embedded PNGs (loss,
   accuracy, eigenvalues) with captions.

The same eigenvalue diagnostic is logged at console-time inside every
analyse run (see `experiments.log_eigenvalue_report`), so spurious
modes are visible in `run.log` too.

---

## Pipeline & module layout

```
neural_dmd/                       library
  config.py                       all knobs
  experiments.py                  hashing, paths, manifest, eigenvalue
                                  reporting, metrics aggregation, summary
  methods.py                      method registry (run callable + label)
  runners.py                      do_analyze / do_plot_loss /
                                  do_plot_accuracy / do_plot_eigenvalues
                                  + run_full_pipeline
  dmdc.py                         method
  sdmdc.py                        method
  optdmdc.py                      method
  coptdmdc.py                     method
  data.py                         dataset registry + DataLoader factory
  model.py                        MLP
  schedule.py                     cosine LR
  metrics.py                      LOSSES, METRICS, loss_at, metric_at
  params.py                       flat <-> tensor parameter conversion
  snapshots.py                    Recorder + .npz schema
  figures.py                      matplotlib helpers
  log.py                          ANSI-coloured logger + log_spectrum
                                  + progress + banner

scripts/                          runnable entry points (thin wrappers)
  train.py                        cache-aware training
  run_exp.py                      full pipeline orchestrator
  analyze_<method>.py             4 thin wrappers -> runners.do_analyze
  plot_loss_<method>.py           4 thin wrappers
  plot_accuracy_<method>.py       4 thin wrappers
  plot_eigenvalues_<method>.py    4 thin wrappers
  test.py                         load model.pt + print test metric
```

Library modules contain no `__main__`; everything runnable lives in
`scripts/`.

---

## Recipes

### The "all" command — full pipeline

```bash
python scripts/run_exp.py
```

Trains if needed, then for every method in `config.METHODS` runs
`analyze + plot_eigenvalues + plot_accuracy + plot_loss`, and aggregates
metrics into `summary.html`.

### Run only some methods

```bash
python scripts/run_exp.py --methods dmdc,sdmdc
python scripts/run_exp.py --methods optdmdc
```

### Run only some operations (`--only`)

```bash
# only analyze + accuracy + eigenvalues (skip the slow loss eval)
python scripts/run_exp.py --only analyze,plot_eigenvalues,plot_accuracy

# shortcut alias for the above
python scripts/run_exp.py --skip-loss

# only the loss plot (analysis must already exist for these methods)
python scripts/run_exp.py --only plot_loss

# minimal: just the analysis npzs, no plots
python scripts/run_exp.py --only analyze
```

Valid `--only` ops: `analyze`, `plot_eigenvalues`, `plot_accuracy`,
`plot_loss`. They run in canonical order regardless of how you list
them.

### Methods × ops combined

```bash
# only DMDc accuracy + eigenvalue plots (assumes analyze already done)
python scripts/run_exp.py --methods dmdc --only plot_accuracy,plot_eigenvalues

# baseline DMDc full pipeline + sDMDc analyze only
python scripts/run_exp.py --methods dmdc                       # all 4 ops for dmdc
python scripts/run_exp.py --methods sdmdc --only analyze       # only analyse sdmdc
```

### Re-run a single method (single-script style)

```bash
python scripts/analyze_optdmdc.py
python scripts/plot_eigenvalues_optdmdc.py
python scripts/plot_accuracy_optdmdc.py
python scripts/plot_loss_optdmdc.py
```

These resolve the active experiment as: env var → latest existing →
fresh.

### Re-run inside a specific historical experiment

```bash
NEURAL_DMD_EXP_ID=2026-05-05_04-08-09_default \
  python scripts/plot_loss_optdmdc.py
```

### Force a fresh training run

```bash
python scripts/run_exp.py --force-train
```

### Compare two configurations side by side

Edit `EXP_LABEL` between runs (e.g. `"plateau-test"` then `"warm-start"`)
and run `run_exp.py` twice. The two experiments live independently
under `outputs/experiments/` and reuse the same training data when the
training-relevant fields match.

---

## How to read the metrics

`summary.html` reports per-method numbers in two tables. Definitions:

### Spectral ρ
Largest eigenvalue magnitude of the reduced operator A. ρ ≤ 1 = forecast
stays bounded; ρ > 1 = forecast grows over time.

### λ>1
Number of eigenvalues with `|λ| > 1 + EIG_STABLE_TOL` (1e-12 default).
The eigenvalue stability section lists each one with its full complex
value, magnitude, and how far above 1.0 it sits.

### In Δacc% / Out Δacc%
**Mean parameter-prediction L2 error**, in percent, over the in-fit /
out-of-fit snapshot grid. Computed as

```
acc_pred(k) = 100 * (1 - ||x̂_k − x_k|| / ||x_k||)
Out Δacc%   = mean over forecast points of |100 − acc_pred(k)|
            = mean of 100 * ||x̂_k − x_k|| / ||x_k||
```

So `Out Δacc% = 16.21` means the forecasted weight vector is, on
average, **16.21% off** from the true weight vector in L2 norm at each
forecast snapshot. This is **NOT** test-accuracy of the predicted
network; it's parameter-vector closeness. For genuine downstream test
accuracy of the predicted network, see "Forecast-region final / min"
below.

### In Δloss / Out Δloss
**Mean absolute test-loss deviation** between predicted and actual
network at the snapshot grid:

```
ΔL_k     = | L(x̂_k; D) − L(x_k; D) |
Out Δloss = mean over forecast points of ΔL_k
```

`L` is the loss in `config.LOSS` (cross-entropy by default), evaluated
on the test set.

### Forecast-region final / min table

When the loss plot ran, this second table compares predicted vs actual
test loss in the forecast window:

| Column            | Meaning                                                          |
|-------------------|------------------------------------------------------------------|
| actual final L    | true network's loss at the LAST snapshot in the forecast window  |
| pred final L      | predicted network's loss at the same point                       |
| actual min L      | minimum of true network's loss over the forecast window          |
| pred min L        | minimum of predicted network's loss over the forecast window     |

`pred final L` close to `actual final L` = the forecast lands at the
right loss. `pred min L` ≪ `actual min L` = the forecast undershoots
(predicts an unrealistically good model at some point, often a
diverging-mode artefact).

The metric we care about for forecasting quality is **parameter
prediction accuracy** (the accuracy plot's y-axis) — how close the
predicted weight vector is to the actual weight vector in L2. The
network's downstream test accuracy is *not* tracked separately; the
`out Δloss` column already captures whether the predicted weights make
a sensible network.

---

## Method details

### DMDc (Proctor, Brunton, Kutz - SIAM J. Appl. Dyn. Syst. 2016)

`x_{k+1} ≈ A x_k + B u_k`. Reduced operator `F = U_x^T A U_x` of size
`p × p` is computed by two SVDs:

1. SVD of `Ω = [X_1; Υ]` → `U_o, Σ_o, V_o`.
2. Independent SVD of `X_fit` → `U_x` (output POD basis).

Then `F = U_x^T X_2 V_o Σ_o^{-1} U_o^T U_x` and `G = U_x^T X_2 V_o
Σ_o^{-1} U_2^T`. Forecast iterates `y ← F y + G u` step-by-step.

### sDMDc

Same as DMDc, plus a single radial-projection of the eigenvalues:

```
gamma_clipped = min(Re(gamma), 0) + Im(gamma) * i
mu_clipped    = exp(gamma_clipped * dt)
F_new         = Z diag(mu_clipped) Z^{-1}
```

No LM. Cheapest stable variant.

### OptDMDc (Askham-Kutz 2018, extended in Rains et al. 2024 §2.3)

Variable projection over the continuous-time eigenvalues `gamma`:

```
H_0^T = Psi(gamma, t) * Omega          where  Omega = Psi^+ H_0^T
min_gamma || H_0^T - Psi(gamma) Omega(gamma) ||_F
```

Solved by Levenberg-Marquardt. The dense Jacobian `J^mat_j` is built
column-by-column from closed-form pieces (paper Eq. 32) and the
augmented LSQ system is recast in real arithmetic for `numpy.linalg.lstsq`.
No stability constraint - eigenvalues can land anywhere in the complex
plane.

### cOptDMDc (Rains et al. JCP 2024 §2.4)

Same as OptDMDc plus

1. Initial radial projection: `gamma <- min(Re(gamma), 0) + Im(gamma) * i`.
2. Linear inequality constraint on the LM update:
   `Re(delta) >= Re(gamma)` so `Re(gamma_new) = Re(gamma) - Re(delta) ≤ 0`.

The constrained LSQ subproblem is recast in real arithmetic and solved
by `scipy.optimize.lsq_linear` with `bounds=(...)`.

---

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# or
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

uv venv
uv pip install -r requirements.txt
```

CUDA is used automatically when available, otherwise CPU.

## References

- Proctor, J. L., Brunton, S. L., & Kutz, J. N. (2016). *Dynamic Mode
  Decomposition with Control*. SIAM J. Appl. Dyn. Syst., 15(1).
- Askham, T., & Kutz, J. N. (2018). *Variable Projection Methods for an
  Optimized DMD*. SIAM J. Appl. Dyn. Syst., 17(1).
- Rains, J., Wang, Y., House, A., Kaminsky, A. L., Tison, N. A., &
  Korivi, V. M. (2024). *Constrained optimized dynamic mode
  decomposition with control for physically stable systems with
  exogeneous inputs*. J. Comput. Phys., 496, 112604.
