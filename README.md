# neural-dmd

Train a small neural network on MNIST, then ask: **can a simple linear
equation predict how the network's parameters evolve during training, well
enough to forecast its loss curve into the future?**

That linear equation is DMDc (Dynamic Mode Decomposition with control). The
project trains the network, records its parameters every few steps, fits
DMDc to that recording, then plots how closely DMDc's prediction tracks the
real training behaviour.

Everything is configured from one file (`config.py`) and built around a few
small registries, so swapping the dataset, the loss function, the network
architecture, or the "control input" handed to DMDc is a single-line change.

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
cosine annealing). We keep one snapshot every 50 steps, so over 5 epochs of
MNIST we end up with about 94 snapshots.

### The loss curve
For any snapshot $x_k$ we can ask: how badly does the network predict on
the test set when its parameters are set to $x_k$? That number is the
**loss** $\mathcal{L}_k = \mathcal{L}(x_k; \mathcal{D})$. Plotting $\mathcal{L}_k$
against $k$ gives the familiar "loss goes down" training curve.

### DMDc — what is it actually doing?
DMDc tries to find two matrices $A$ and $B$ such that the entire trajectory
above is well-approximated by a single linear rule:
$$x_{k+1} \;\approx\; A\,x_k \;+\; B\,u_k$$
In words: "the next parameter vector is mostly a linear function of the
current parameter vector plus the learning rate." This is a strong
assumption — real neural-net training is not literally linear — but it
turns out to be a surprisingly good local approximation.

DMDc fits $A$ and $B$ from the recorded snapshots using a standard
truncated-SVD routine (Proctor, Brunton, Kutz 2016). Because $x_k$ has
109,000 entries we never form $A$ in full — instead we work in a much
smaller "principal-component" subspace and only lift back to the full
parameter space when we need to.

### What we then check
Once $A, B$ are fit, we can roll the rule forward starting from $x_0$ to
get a **predicted** trajectory $\hat{x}_0, \hat{x}_1, \hat{x}_2, \dots$.
Two things are then interesting:

1. **Loss curve comparison** (`dmdc_loss.png`). Plug each $\hat{x}_k$ back
   into the network and compute its loss — same way as for the real $x_k$.
   The two loss curves should overlap inside the fit region (DMDc has seen
   that data) and possibly diverge outside it (DMDc is now extrapolating).

2. **Parameter-prediction-accuracy** (`dmdc_accuracy.png`). How close is
   $\hat{x}_k$ to the real $x_k$ as a vector? Plot
   $100 \cdot \bigl(1 - \|\hat{x}_k - x_k\|_2 / \|x_k\|_2\bigr)$ against
   $k$. By construction it sits at ~100% inside the fit region; outside it
   shows how the linear forecast slowly drifts away from the truth.

The two plots are complementary: the loss plot is what a practitioner
cares about; the accuracy plot is the honest measure of the linear model's
forecast quality in parameter space.

---

## Files

```
config.py     ← single source of truth for every knob
data.py       ← dataset registry + DataLoader factory
model.py      ← the MLP (layer sizes from config.ARCH)
schedule.py   ← cosine-annealing learning-rate scheduler
metrics.py    ← loss + metric registries, plus loss_at / metric_at evaluators
params.py     ← flatten / unflatten model parameters ↔ flat vector
snapshots.py  ← Recorder that captures (x_k, u_k) during training
dmdc.py       ← truncated-SVD DMDc fit + iterative forecast (pure NumPy)
analysis.py   ← load_snapshots + fit_and_forecast helpers shared by plots
figures.py    ← shared matplotlib helper for the two comparison plots
log.py        ← tiny ANSI-coloured console logger
train.py      ← runs training, writes model.pt + snapshots.npz
plot.py       ← writes dmdc_loss.png  (Equation 2: loss curves)
eval.py       ← prints test accuracy + writes dmdc_accuracy.png
```

### What each file does

- **`config.py`** — every tunable lives here: which dataset, which loss /
  metric, network architecture, learning rate, snapshot interval, DMDc
  rank and fit fraction, and a `control_fn(optimizer, step)` callback that
  defines what the "control input" $u_k$ is. Edit values; nothing else
  needs to change.
- **`data.py`** — exposes `get_loaders(name, ...)` and a `DATASETS`
  registry. MNIST, Fashion-MNIST and CIFAR-10 ship out of the box; add a
  new entry to use a different one.
- **`model.py`** — the MLP, parameterised by a list of layer sizes. Same
  module works for any flat-input classification task; just adjust the
  list to match the dataset's input dim and class count.
- **`schedule.py`** — cosine-annealing learning-rate schedule that decays
  smoothly from `LR` down to zero across the whole run.
- **`metrics.py`** — `LOSSES` and `METRICS` registries plus two helpers
  `loss_at(net, vec, ...)` / `metric_at(net, vec, ...)` that load a
  parameter vector into the network and evaluate it on a DataLoader.
- **`params.py`** — `flatten_params(net)` packs every weight and bias into
  one 1-D NumPy array; `load_params(net, vec)` is the inverse. This is how
  we turn the network into a vector $x$ for DMDc.
- **`snapshots.py`** — the `Recorder` class. Plug it into the training
  loop; it saves `(x_k, u_k)` pairs every N gradient steps and writes them
  to a `.npz` at the end. The control input is whatever `control_fn`
  returns, so this module knows nothing about learning rates specifically.
- **`dmdc.py`** — pure NumPy DMDc. `fit_dmdc(X, U, rank)` returns reduced
  matrices $A$ and $B$ along with a POD basis; `forecast(model, x0, U_seq)`
  iterates the rule forward and lifts back to full state space. About 50
  lines, well-commented.
- **`analysis.py`** — small wrapper used by both plot scripts:
  `load_snapshots(path)` and `fit_and_forecast(X, U, frac, rank)`.
- **`figures.py`** — `comparison_plot(...)` draws the two-curve dark-themed
  figure used by both `plot.py` and `eval.py` so the visual style is
  consistent.
- **`log.py`** — coloured tagged console output:
  `INFO` (cyan), `OK` (green), `WARN` (yellow), `ERR` (red).
- **`train.py`** — the training entry point. Builds the loaders / model /
  optimizer / scheduler / recorder, runs the loop, writes `model.pt` and
  `snapshots.npz`.
- **`plot.py`** — loads the snapshots, fits DMDc, evaluates the
  configurable loss on real and forecast parameters, writes
  `dmdc_loss.png`.
- **`eval.py`** — prints the trained model's metric on the test set and
  writes `dmdc_accuracy.png` (parameter-prediction accuracy with the
  formula $e(k) = \|\hat{x}_k - x_k\|_2 / \|x_k\|_2$).

### How they fit together

```
config.py (edit knobs)
    │
    ▼
train.py ─────────────► model.pt
    │                   snapshots.npz
    │  (uses)
    ├── data.py · model.py · metrics.py · schedule.py · snapshots.py · params.py · log.py
    │
    │
    ├──► plot.py ─────► dmdc_loss.png
    │       │
    │       └── analysis.py · dmdc.py · figures.py · metrics.py · data.py · model.py
    │
    └──► eval.py ─────► dmdc_accuracy.png  (+ console: trained-model metric)
            │
            └── analysis.py · dmdc.py · figures.py · metrics.py · data.py · model.py
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

```bash
uv run python train.py     # trains, writes model.pt + snapshots.npz
uv run python plot.py      # writes dmdc_loss.png
uv run python eval.py      # prints test metric + writes dmdc_accuracy.png
```

First run downloads the dataset to `./data/`. Uses CUDA if available, else
CPU.

---

## Output

### Train

```
INFO  dataset=mnist  arch=[784, 128, 64, 10]  device=cuda  epochs=5  lr=0.001  loss=cross_entropy  snap_every=50
OK    epoch 1/5  train_loss=0.2710  train_acc=91.98%  lr=9.05e-04
...
INFO  saved weights to model.pt
INFO  saved 94 snapshots to snapshots.npz  (run plot.py and eval.py for DMDc analysis)
```

### plot.py — DMDc loss curve

`dmdc_loss.png` — actual loss vs DMDc-predicted loss. They overlap inside
the fit region (shaded) and the DMDc curve continues to track the real
loss closely into the forecast region because the parameters are by then
in a flat plateau near the minimum.

### eval.py — DMDc parameter-prediction accuracy

`dmdc_accuracy.png` — the formula $e(k) = \|\hat{x}_k - x_k\|_2 / \|x_k\|_2$
expressed as accuracy $100\,(1-e(k))\%$. The "actual" curve is a flat
reference at 100% (a vector compared against itself); the DMDc curve sits
at ~100% inside the fit region and drops in the forecast region as small
one-step prediction errors compound through the iteration.

```
INFO  loading model.pt on cuda
OK    trained-model accuracy = 97.89%
INFO  loaded snapshots.npz  X=(109386, 94)  U=(1, 93)  m=94 snapshots
OK    DMDc fit on first 47/94 snapshots, rank=46
OK    saved figure to dmdc_accuracy.png
```

---

## Plug-and-play extension points

| What you want to change | Edit                                    |
|-------------------------|-----------------------------------------|
| Dataset                 | `config.DATASET` (must be in `data.DATASETS`) — add one if needed; update `config.ARCH` |
| Network architecture    | `config.ARCH` (list of MLP layer widths) |
| Loss function           | Add to `metrics.LOSSES`, set `config.LOSS` |
| Test metric             | Add to `metrics.METRICS`, set `config.METRIC` |
| Control input $u_k$     | Edit `config.control_fn(optimizer, step)` to return any iterable of floats |
| Snapshot frequency      | `config.SNAP_EVERY` |
| DMDc fit fraction       | `config.FIT_FRAC` (0.0–1.0) |
| DMDc rank truncation    | `config.RANK` (`None` = full available; small int to study compression) |
| Epochs / LR             | `config.EPOCHS`, `config.LR` |
