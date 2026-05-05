# neural-dmd

Train a small neural network on MNIST, then ask: **can a simple linear
equation predict how the network's parameters evolve during training, well
enough to forecast its loss curve into the future?**

That linear equation is DMDc (Dynamic Mode Decomposition with control). The
project trains the network, records its parameters during training, fits
DMDc to that recording, then plots how closely DMDc's prediction tracks the
real training behaviour.

Everything is configured from one file (`neural_dmd/config.py`) and built
around a few small registries, so swapping the dataset, the loss function,
the network architecture, or the "control input" handed to DMDc is a
single-line change.

---

## What is going on, in plain language

### The neural network
A neural network is a big function defined by a long list of numbers called
**parameters** (weights and biases). For our network — a simple
fully-connected MLP with layer sizes 784 → 128 → 64 → 10 — there are about
**109,000** of them. Stack all of them into one tall column and you get a
single vector $x \in \mathbb{R}^{109386}$ that completely describes the
network at one moment in time.

### Training
"Training" means repeatedly nudging $x$ in directions that make the network
better at classifying digits. After every nudge (one **gradient step**), $x$
is slightly different. So training produces a **trajectory**:
$$x_0 \to x_1 \to x_2 \to \dots$$
Each $x_k$ is a 109,386-dimensional snapshot of the network. The size of
the nudge is set by the **learning rate** $u_k$ (here decayed smoothly via
cosine annealing). With batch size 100 there are exactly 600 gradient
steps per epoch, so 5 epochs cover **3,000 steps total**.

### Two snapshot cadences (this is the interesting bit)
The recorder uses **two different sampling rates** for the parameter
trajectory:

* **Fit region** (the first half of training, steps 0…1499): record
  **every single step**. That's 1,500 dense snapshots — the data DMDc
  will fit its linear model on.
* **Forecast region** (the second half, steps 1500…2999): record **every
  50th step**. That's 30 sparse comparison points — we just need them
  to score how good DMDc's forecast is, not to fit on.

Why split it? DMDc only sees the fit-region data. More fit data → a
better-conditioned linear model → a more accurate forecast. Snapshotting
every step in the fit region gives the algorithm 30× more transitions to
learn from than a uniform every-50 cadence. The forecast region stays
sparse because we just need somewhere to **compare** the predicted
trajectory to the truth, and evaluating loss at thousands of points
would be expensive for no extra information.

The full per-step learning-rate sequence is also recorded (regardless of
whether a snapshot was taken at that step) so DMDc has a control value
$u_k$ for every step it iterates through during the forecast.

### The loss curve
For any snapshot $x_k$ we can ask: how badly does the network predict on
the test set when its parameters are set to $x_k$? That number is the
**loss** $\mathcal{L}_k = \mathcal{L}(x_k; \mathcal{D})$. Plotting
$\mathcal{L}_k$ against $k$ gives the familiar "loss goes down" training
curve.

### DMDc — what is it actually doing?
DMDc tries to find two matrices $A$ and $B$ such that the entire
fit-region trajectory is well-approximated by a single linear rule:
$$x_{k+1} \;\approx\; A\,x_k \;+\; B\,u_k$$
In words: "the next parameter vector is mostly a linear function of the
current parameter vector plus the learning rate." This is a strong
assumption — real neural-net training is not literally linear — but it
turns out to be a surprisingly good local approximation.

DMDc fits $A$ and $B$ from the dense fit-region data using a standard
truncated-SVD routine (Proctor, Brunton, Kutz 2016). Because $x_k$ has
109,000 entries we never form $A$ in full — instead we work in a much
smaller "principal-component" subspace of dimension at most
$\min(n,\text{fit\_steps})=1{,}500$ and only lift back to the full
parameter space when we need to. The SVD runs in f64 internally for
numerical stability (the parameter trajectory has a long tail of small
singular values).

### What we then check
Once $A, B$ are fit, we roll the rule forward starting from
$x_{1499}$ (the last fit snapshot) to predict $\hat{x}_{1500}, \hat{x}_{1501},
\dots, \hat{x}_{2999}$. The forecast is sampled at the same gradient-step
indices where we have ground-truth comparison snapshots (every 50th step
in the forecast region). Two things are then interesting:

1. **Loss curve comparison** (`dmdc_loss.png`). Plug each $\hat{x}_k$
   back into the network and compute its loss. The two loss curves
   should overlap inside the fit region (DMDc has seen that data) and
   may diverge outside it (DMDc is now extrapolating).

2. **Parameter-prediction-accuracy** (`dmdc_accuracy.png`). How close is
   $\hat{x}_k$ to the real $x_k$ as a vector? Plot
   $100 \cdot \bigl(1 - \|\hat{x}_k - x_k\|_2 / \|x_k\|_2\bigr)$ against
   $k$. By construction it sits at ~100% inside the fit region (full-rank
   in-sample reconstruction is exact); outside it shows how the linear
   forecast slowly drifts away from the truth.

The two plots are complementary: the loss plot is what a practitioner
cares about; the accuracy plot is the honest measure of the linear
model's forecast quality in parameter space.

---

## Numbers cheat sheet

| Quantity                              | Value                              |
|---------------------------------------|------------------------------------|
| Dataset                               | MNIST (60k train / 10k test)       |
| Batch size                            | 100 (60000 / 100 = 600 steps/epoch exactly) |
| Epochs                                | 5                                  |
| Total gradient steps                  | 3,000                              |
| Parameter count $n$                   | 109,386                            |
| Snap cadence — fit region             | every gradient step                |
| Snap cadence — forecast region        | every 50 gradient steps            |
| Fit / forecast split                  | 1,500 / 1,500 gradient steps       |
| Snapshots used to fit DMDc            | 1,500                              |
| Snapshots forecast & compared         | 30                                 |
| Plot evaluation grid                  | every 50 steps  →  60 plot points  |
| `snapshots.npz` size on disk          | ~640 MB                            |

---

## Project layout

```
neural-dmd/
├── README.md
├── requirements.txt
├── neural_dmd/                  ← library code (importable package)
│   ├── __init__.py
│   ├── config.py                single source of truth for every knob + paths
│   ├── data.py                  dataset registry + DataLoader factory
│   ├── model.py                 the MLP (layer sizes from config.ARCH)
│   ├── schedule.py              cosine LR scheduler + closed-form cosine_lr_at()
│   ├── metrics.py               loss + metric registries, plus loss_at / metric_at
│   ├── params.py                flatten / unflatten model parameters ↔ flat vector
│   ├── snapshots.py             dual-cadence Recorder + Recorder.load + eval_indices
│   ├── dmdc.py                  DMDc fit + forecast (NumPy, f64 internals)
│   ├── figures.py               matplotlib helpers: comparison_plot + eigenvalue_plot
│   └── log.py                   ANSI-coloured logger + banner / progress helpers
├── scripts/                     ← runnable entry points
│   ├── train.py                 trains the network → outputs/model.pt + snapshots.npz
│   ├── analyze.py               runs DMDc once → outputs/analysis.npz (cached)
│   ├── test.py                  prints the trained network's test metric
│   ├── plot_loss.py             outputs/plots/dmdc_loss.png
│   ├── plot_accuracy.py         outputs/plots/dmdc_accuracy.png
│   └── plot_eigenvalues.py      outputs/plots/dmdc_eigenvalues.png
├── data/                        ← input dataset cache (MNIST etc., downloaded on demand)
└── outputs/                     ← all generated artifacts (created automatically)
    ├── model.pt
    ├── snapshots.npz
    ├── analysis.npz
    └── plots/
        ├── dmdc_loss.png
        ├── dmdc_accuracy.png
        └── dmdc_eigenvalues.png
```

Library code lives in `neural_dmd/`; nothing in there has a `__main__`.
Runnable scripts live in `scripts/`; each one starts with a 4-line
`sys.path` bootstrap so it can be invoked directly with `python scripts/X.py`
from anywhere on the filesystem. All paths in `config.py` are anchored
to the project root, so generated files always land under `outputs/`
regardless of where you launched from.

### What each file does

- **`config.py`** — every tunable lives here: dataset, loss / metric,
  network architecture, learning rate, the two snap cadences
  (`SNAP_FIT_EVERY`, `SNAP_FORECAST_EVERY`), DMDc rank, fit fraction,
  and the `control_fn(optimizer, step)` callback that defines the
  control input $u_k$. Edit values; nothing else needs to change.
- **`data.py`** — exposes `get_loaders(name, ...)` and a `DATASETS`
  registry. MNIST, Fashion-MNIST and CIFAR-10 ship out of the box.
- **`model.py`** — the MLP, parameterised by a list of layer sizes.
- **`schedule.py`** — the PyTorch cosine scheduler used during training,
  plus a pure `cosine_lr_at(step, total_steps, lr_max, min_lr)` function
  that returns the LR at any step without an optimizer.
- **`metrics.py`** — `LOSSES` and `METRICS` registries plus
  `loss_at(net, vec, ...)` / `metric_at(net, vec, ...)` evaluators that
  load a parameter vector into the network and run a forward pass.
- **`params.py`** — `flatten_params(net)` packs every weight and bias into
  one 1-D NumPy array; `load_params(net, vec)` is the inverse.
- **`snapshots.py`** — the dual-cadence `Recorder`. Records `(x_k, u_k)`
  pairs at one rate inside `[0, fit_steps)` and at another rate inside
  `[fit_steps, total_steps)`, while always recording $u_k$ for *every*
  gradient step. Saves an `.npz` with `X`, `U`, `steps`, plus the
  scalars `fit_split`, `fit_steps`, `total_steps`, `fit_every`,
  `forecast_every`.
- **`analysis.py`** — `load_snapshots(path)` returns the snapshot bundle
  as a dict; `fit_and_forecast(snap, rank)` runs DMDc; `eval_indices`
  is the tiny helper plot.py / eval.py use to subsample the dense
  fit-region snapshots down to the same per-50 grid as the forecast
  region.
- **`dmdc.py`** — the DMDc fit + forecast in pure NumPy. Cast inputs to
  f64 internally for SVD stability; output X_pred is cast back to f32
  to match the snapshot NPZ schema.
- **`figures.py`** — `comparison_plot(...)` draws the dark-themed
  two-curve figure used by both `plot.py` and `eval.py`.
- **`log.py`** — coloured tagged console output (`INFO` cyan, `OK`
  green, `WARN` yellow, `ERR` red) plus `banner(label, **fields)` and
  `progress(label, current, total, t_start)` helpers used everywhere.
- **`train.py`** — training entry point. Builds loaders / model /
  optimizer / scheduler / dual-cadence recorder, runs the loop, writes
  `model.pt` and `snapshots.npz`. Logs intra-epoch progress every 25%.
- **`plot.py`** — loads the snapshots, runs DMDc, evaluates the
  configurable loss on real and forecast parameters at the per-50
  evaluation grid, writes `dmdc_loss.png`.
- **`eval.py`** — prints the trained model's metric, runs DMDc,
  computes parameter-prediction accuracy at the per-50 evaluation
  grid, writes `dmdc_accuracy.png`.

### How they fit together

```
neural_dmd/config.py (edit knobs)
        │
        ▼
scripts/train.py ───────► outputs/model.pt
                          outputs/snapshots.npz   (1500 dense + 30 sparse snapshots)
                                  │
                                  ▼
                          scripts/analyze.py ───► outputs/analysis.npz
                                                    (X_pred + A's eigenvalues)
                                  │
        ┌──────────────┬──────────┴──────────┬──────────────────────────┐
        ▼              ▼                     ▼                          ▼
  scripts/test.py  scripts/plot_loss.py  scripts/plot_accuracy.py  scripts/plot_eigenvalues.py
        │              │                     │                          │
   trained-       outputs/plots/        outputs/plots/             outputs/plots/
   model          dmdc_loss.png         dmdc_accuracy.png          dmdc_eigenvalues.png
   metric
   (stdout)
```

MNIST ships a fixed 60k/10k train/test split. Train and eval never share
samples.

---

## Install uv

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Install deps

```bash
uv venv
uv pip install -r requirements.txt
```

## Run

Full pipeline, in order:

```bash
uv run python scripts/train.py            && \
uv run python scripts/analyze.py          && \
uv run python scripts/test.py             && \
uv run python scripts/plot_loss.py        && \
uv run python scripts/plot_accuracy.py    && \
uv run python scripts/plot_eigenvalues.py
```

Or step-by-step:

| Step                          | Reads                                   | Writes / prints                                |
|-------------------------------|-----------------------------------------|------------------------------------------------|
| `scripts/train.py`            | (downloads MNIST into `data/`)          | `outputs/model.pt`, `outputs/snapshots.npz`    |
| `scripts/analyze.py`          | `outputs/snapshots.npz`                 | `outputs/analysis.npz`  (X_pred + eigenvalues) |
| `scripts/test.py`             | `outputs/model.pt`                      | trained-model accuracy printed to stdout       |
| `scripts/plot_loss.py`        | `outputs/snapshots.npz`, `analysis.npz` | `outputs/plots/dmdc_loss.png`                  |
| `scripts/plot_accuracy.py`    | `outputs/snapshots.npz`, `analysis.npz` | `outputs/plots/dmdc_accuracy.png`              |
| `scripts/plot_eigenvalues.py` | `outputs/analysis.npz`                  | `outputs/plots/dmdc_eigenvalues.png`           |

First run downloads MNIST to `data/`. Uses CUDA if available, else CPU.
`outputs/` and `outputs/plots/` are created automatically on first run.
The plot scripts can be re-run independently — only `analyze.py` is
expensive (the SVDs run once and the result is cached).

---

## Output

### Train

```
INFO  == training start  dataset=mnist  arch=[784, 128, 64, 10]  device=cuda  epochs=5  batch=100  lr=0.001  loss=cross_entropy
INFO  trajectory plan: total_steps=3000  fit_steps=1500  snap_fit_every=1 -> ~1500 fit snaps  snap_forecast_every=50 -> ~30 forecast snaps  ~669MB on disk
INFO    epoch 1/5 batches: 150/600 ( 25.0%)  elapsed=   1.1s  eta=   3.2s
INFO    epoch 1/5 batches: 300/600 ( 50.0%)  elapsed=   1.9s  eta=   1.9s
INFO    epoch 1/5 batches: 450/600 ( 75.0%)  elapsed=   2.6s  eta=   0.9s
INFO    epoch 1/5 batches: 600/600 (100.0%)  elapsed=   3.4s  eta=   0.0s
OK    epoch 1/5  train_loss=0.2958  train_acc=91.39%  lr=9.05e-04  snapshots_so_far=600
...
INFO  == training done  total_seconds=17.5  snapshots_taken=1530
OK    saved 1530 snapshots to snapshots.npz  (now run plot.py and eval.py for DMDc analysis)
```

### plot_loss.py — `dmdc_loss.png`

Actual loss vs DMDc-predicted loss at the 60-point per-50-step
evaluation grid. They overlap inside the fit region (shaded) and the
DMDc curve continues to track the real loss closely into the forecast
region because the parameters are by then in a flat plateau near the
minimum.

### plot_accuracy.py — `dmdc_accuracy.png`

The formula $e(k) = \|\hat{x}_k - x_k\|_2 / \|x_k\|_2$ expressed as
accuracy $100\,(1-e(k))\%$. The "actual" curve is a flat reference at
100%; the DMDc curve sits at ~100% inside the fit region and drops in
the forecast region as small one-step prediction errors compound through
the iteration.

### plot_eigenvalues.py — `dmdc_eigenvalues.png`

Each eigenvalue $\lambda$ of the DMDc operator $A$ as a dot in the
complex plane (real part on the x-axis, imaginary part on the y-axis).
The dashed circle is the unit circle $|\lambda| = 1$ — the stability
boundary for a discrete-time linear system. Dots inside the circle are
modes that **decay** as the forecast iterates forward; dots on the
circle **oscillate** without growing; dots outside the circle **grow**
and are responsible for forecast blow-up over long horizons. For a
well-fit DMDc model on training trajectories that converge, you expect
nearly all eigenvalues just inside or on the unit circle, with at most
a handful (often a single one) marginally outside.

---

## Plug-and-play extension points

| What you want to change         | Edit                                                                 |
|---------------------------------|----------------------------------------------------------------------|
| Dataset                         | `config.DATASET` (must be in `data.DATASETS`); update `config.ARCH`  |
| Network architecture            | `config.ARCH` (list of MLP layer widths)                             |
| Loss function                   | Add to `metrics.LOSSES`, set `config.LOSS`                           |
| Test metric                     | Add to `metrics.METRICS`, set `config.METRIC`                        |
| Control input $u_k$             | Edit `config.control_fn(optimizer, step)`                            |
| Fit-region snapshot cadence     | `config.SNAP_FIT_EVERY` (1 = every step)                             |
| Forecast-region snapshot cadence| `config.SNAP_FORECAST_EVERY` (also drives the plot eval grid)        |
| DMDc fit fraction               | `config.FIT_FRAC` (0.0–1.0 of total training steps)                  |
| DMDc rank truncation            | `config.RANK` (`None` = full available; small int to study compression) |
| Epochs / LR                     | `config.EPOCHS`, `config.LR` (`config.LR_MIN` for cosine floor)      |
