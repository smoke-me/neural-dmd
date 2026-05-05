# neural-dmd

Train an MLP on MNIST, record its parameter trajectory during training, fit
a Dynamic Mode Decomposition with control (DMDc) linear model to that
trajectory, and measure the linear model's ability to forecast the trained
network's behavior over portions of the training run not used for fitting.

The full configuration lives in `neural_dmd/config.py`. All swappable
components (dataset, model architecture, loss, evaluation metric, control
input, snapshot cadences, DMDc rank, fit/forecast split) are exposed there
as single-line edits.

## Method

### Parameterization

The network's weights and biases are flattened into a single state vector
$x \in \mathbb{R}^n$, where $n$ is the total parameter count. For the default
architecture (`config.ARCH = [784, 128, 64, 10]`), $n = 109{,}386$. Each
gradient update produces a new state vector, so a training run of $T$ steps
yields a trajectory $x_0, x_1, \dots, x_T$. The learning rate at step $k$,
denoted $u_k$, is treated as a scalar control input; the default schedule
is cosine annealing.

### Snapshot capture

The training run is partitioned into a *fit region* $[0, T_\mathrm{fit})$
and a *forecast region* $[T_\mathrm{fit}, T)$ by `config.FIT_FRAC`. Within
each region the recorder samples the parameter trajectory at an independent
cadence (`config.SNAP_FIT_EVERY` and `config.SNAP_FORECAST_EVERY`). The
fit-region samples form the dataset DMDc fits its linear model to; the
forecast-region samples serve as ground-truth comparison points for the
predicted trajectory. The control sequence $u_k$ is recorded at every
gradient step regardless of snapshot cadence, since the forecast iteration
requires a control value at each step it advances through.

### DMDc fit

DMDc estimates matrices $A$ and $B$ satisfying

$$x_{k+1} \approx A\,x_k + B\,u_k$$

from the fit-region snapshots. The full operator $A \in \mathbb{R}^{n \times n}$
is never formed. Instead, the algorithm of Proctor, Brunton, and Kutz
(SIAM J. Appl. Dyn. Syst., 2016) projects the dynamics onto a
low-dimensional POD basis derived from the snapshot data, with reduced
dimension at most $\min(n, m_\mathrm{fit})$, where $m_\mathrm{fit}$ is the
number of fit-region snapshots. The SVD runs in `float64` internally so
that singular values close to machine precision are resolved without
loss of orthogonality; the predicted trajectory is cast back to `float32`
on output to match the snapshot file schema.

### Forecast and evaluation

Once $(A, B)$ are fit, the linear rule is iterated forward from the final
fit-region snapshot, applying the recorded control sequence step by step,
to produce a predicted trajectory $\hat{x}_k$ over the forecast region.
Three figures and one stdout report compare the prediction to ground truth:

1. `dmdc_loss.png` overlays the test-set loss $\mathcal{L}(x_k; \mathcal{D})$
   against $\mathcal{L}(\hat{x}_k; \mathcal{D})$ across the trajectory.
2. `dmdc_accuracy.png` plots the parameter-prediction accuracy
   $100\,(1 - \|\hat{x}_k - x_k\|_2 / \|x_k\|_2)\,\%$.
3. `dmdc_eigenvalues.png` shows the eigenvalues of the reduced operator
   $A$ in the complex plane against the unit circle, which is the
   stability boundary for a discrete-time linear system.
4. `scripts/test.py` reports the trained model's chosen metric (default:
   classification accuracy) on the held-out test set.

## Default configuration

| Parameter                       | Value                                              |
|---------------------------------|----------------------------------------------------|
| Dataset                         | MNIST (60,000 train / 10,000 test)                 |
| Batch size                      | 100                                                |
| Epochs                          | 5                                                  |
| Total gradient steps            | 3,000                                              |
| Architecture                    | MLP 784 → 128 → 64 → 10                            |
| Parameter count $n$             | 109,386                                            |
| Learning rate schedule          | Cosine annealing, $\eta_0 = 10^{-3}$, $\eta_\min = 0$ |
| Loss                            | Cross-entropy                                      |
| Fit / forecast split            | 1,500 / 1,500 gradient steps                       |
| `SNAP_FIT_EVERY`                | 1                                                  |
| `SNAP_FORECAST_EVERY`           | 50                                                 |
| Fit-region snapshots            | 1,500                                              |
| Forecast-region snapshots       | 30                                                 |
| DMDc rank                       | Full available, $\min(n, m_\mathrm{fit}) = 1{,}500$ |
| `outputs/snapshots.npz` size    | ≈ 640 MB                                           |

## Project layout

```
neural-dmd/
├── README.md
├── requirements.txt
├── neural_dmd/                  # library (importable package)
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── dmdc.py
│   ├── figures.py
│   ├── log.py
│   ├── metrics.py
│   ├── model.py
│   ├── params.py
│   ├── schedule.py
│   └── snapshots.py
├── scripts/                     # runnable entry points
│   ├── train.py
│   ├── analyze.py
│   ├── test.py
│   ├── plot_loss.py
│   ├── plot_accuracy.py
│   └── plot_eigenvalues.py
├── data/                        # downloaded dataset cache
└── outputs/                     # generated artifacts (created automatically)
    ├── model.pt
    ├── snapshots.npz
    ├── analysis.npz
    └── plots/
        ├── dmdc_loss.png
        ├── dmdc_accuracy.png
        └── dmdc_eigenvalues.png
```

Library modules contain no `__main__`; all runnable code lives in
`scripts/`. Each script begins with a short bootstrap that adds the project
root to `sys.path`, allowing direct invocation (`python scripts/X.py`) from
any working directory. All output paths in `config.py` are anchored to the
project root.

## Module reference

| Module | Responsibility |
|---|---|
| `config.py`              | Centralized configuration: hyperparameters, snapshot cadences, file paths, and the `control_fn(optimizer, step)` callback producing the DMDc control input. |
| `data.py`                | Dataset registry (`DATASETS`) and DataLoader factory (`get_loaders`). MNIST, Fashion-MNIST, and CIFAR-10 are pre-registered. |
| `model.py`               | MLP parameterized by a list of layer widths. |
| `schedule.py`            | Cosine annealing scheduler used during training, plus a closed-form `cosine_lr_at(step, total_steps, lr_max, min_lr)`. |
| `metrics.py`             | `LOSSES` and `METRICS` registries; `loss_at` / `metric_at` evaluate a parameter vector against a DataLoader. |
| `params.py`              | Bidirectional conversion between the network's named parameter tensors and a single flat NumPy vector. |
| `snapshots.py`           | Dual-cadence `Recorder`, its on-disk schema (`save`, `load`), and `eval_indices(steps, every)` for subsampling. |
| `dmdc.py`                | DMDc fit and forecast in NumPy. Returns the reduced operator $A$, the modes, and the predicted trajectory. |
| `figures.py`             | Two matplotlib helpers: `comparison_plot` (loss / accuracy curves) and `eigenvalue_plot`. |
| `log.py`                 | ANSI-colored console logger with `INFO` / `OK` / `WARN` / `ERR` tags, plus `banner` and `progress` helpers. |
| `scripts/train.py`       | Trains the model; writes `outputs/model.pt` and `outputs/snapshots.npz`. |
| `scripts/analyze.py`     | Runs DMDc once, computes eigenvalues of $A$; writes `outputs/analysis.npz`. |
| `scripts/test.py`        | Loads the trained model; prints its test-set metric to stdout. |
| `scripts/plot_loss.py`   | Writes `outputs/plots/dmdc_loss.png`. |
| `scripts/plot_accuracy.py`     | Writes `outputs/plots/dmdc_accuracy.png`. |
| `scripts/plot_eigenvalues.py`  | Writes `outputs/plots/dmdc_eigenvalues.png`. |

## Pipeline

```
neural_dmd/config.py
        │
        ▼
scripts/train.py ─────────► outputs/model.pt
                            outputs/snapshots.npz
                                    │
                                    ▼
                            scripts/analyze.py ───► outputs/analysis.npz
                                    │
        ┌────────────────┬──────────┴───────────┬────────────────────────────┐
        ▼                ▼                      ▼                            ▼
  scripts/test.py  scripts/plot_loss.py  scripts/plot_accuracy.py  scripts/plot_eigenvalues.py
        │                │                      │                            │
     stdout         outputs/plots/         outputs/plots/               outputs/plots/
                    dmdc_loss.png          dmdc_accuracy.png            dmdc_eigenvalues.png
```

| Step                          | Inputs                                       | Outputs                                              |
|-------------------------------|----------------------------------------------|------------------------------------------------------|
| `scripts/train.py`            | (downloads MNIST into `data/`)               | `outputs/model.pt`, `outputs/snapshots.npz`          |
| `scripts/analyze.py`          | `outputs/snapshots.npz`                      | `outputs/analysis.npz`                               |
| `scripts/test.py`             | `outputs/model.pt`                           | trained-model metric (stdout)                        |
| `scripts/plot_loss.py`        | `outputs/snapshots.npz`, `outputs/analysis.npz` | `outputs/plots/dmdc_loss.png`                     |
| `scripts/plot_accuracy.py`    | `outputs/snapshots.npz`, `outputs/analysis.npz` | `outputs/plots/dmdc_accuracy.png`                 |
| `scripts/plot_eigenvalues.py` | `outputs/analysis.npz`                       | `outputs/plots/dmdc_eigenvalues.png`                 |

`analyze.py` performs the only computationally significant work in the
analysis pipeline (the SVDs). The plot scripts consume its cached output
and may be re-run independently.

## Installation

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Create the environment and install dependencies:

```bash
uv venv
uv pip install -r requirements.txt
```

## Running

Full pipeline:

```bash
uv run python scripts/train.py            && \
uv run python scripts/analyze.py          && \
uv run python scripts/test.py             && \
uv run python scripts/plot_loss.py        && \
uv run python scripts/plot_accuracy.py    && \
uv run python scripts/plot_eigenvalues.py
```

The first invocation of `train.py` downloads MNIST into `data/`. CUDA is
used automatically when available, otherwise the CPU. The `outputs/` and
`outputs/plots/` directories are created on first run.

## Extension points

| To change                         | Edit                                                                  |
|-----------------------------------|-----------------------------------------------------------------------|
| Dataset                           | `config.DATASET` (must be registered in `data.DATASETS`); update `config.ARCH` accordingly. |
| Network architecture              | `config.ARCH` (list of MLP layer widths).                             |
| Loss function                     | Add to `metrics.LOSSES`, then set `config.LOSS`.                      |
| Evaluation metric                 | Add to `metrics.METRICS`, then set `config.METRIC`.                   |
| Control input $u_k$               | `config.control_fn(optimizer, step)`.                                 |
| Fit-region snapshot cadence       | `config.SNAP_FIT_EVERY`.                                              |
| Forecast-region snapshot cadence  | `config.SNAP_FORECAST_EVERY`.                                         |
| Fit / forecast split              | `config.FIT_FRAC` ($\in [0, 1]$).                                     |
| DMDc rank truncation              | `config.RANK` (`None` = full available; integer for explicit truncation). |
| Epochs and learning rate          | `config.EPOCHS`, `config.LR`, `config.LR_MIN`.                        |
