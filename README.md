# neural-dmd

MNIST training: 784 → 128 → 64 → 10 MLP in PyTorch.

## Files

- `model.py` — `MLP` network definition.
- `data.py` — MNIST download, transforms, train/test `DataLoader`s.
- `train.py` — train loop + entry point. Saves weights to `model.pt` at end. Hyperparameters at top.
- `eval.py` — loads `model.pt` and reports test accuracy. Also exports `evaluate()` reused by `train.py`.
- `schedule.py` — learning-rate schedule factory. Currently cosine annealing, swappable in one place.
- `log.py` — tiny ANSI-colored console logger. `log(tag, msg)` with tags `info` (cyan), `ok` (green), `warn` (yellow), `err` (red).
- `requirements.txt` — `torch`, `torchvision`.

MNIST ships a fixed 60k/10k train/test split. Train and eval never share samples.

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

## Train

```bash
uv run python train.py
```

First run downloads MNIST to `./data/`. Uses CUDA if available, else CPU. Writes `model.pt` when done.

## Eval

```bash
uv run python eval.py
```

Loads `model.pt` and prints test accuracy on the 10k held-out test set.

## Output

Train (numbers measured on the training set - shows the model fitting). `lr` decays each epoch via cosine annealing:

```
INFO  device=cuda  epochs=5  lr=0.001  schedule=cosine
OK    epoch 1/5  train_loss=0.2543  train_acc=92.61%  lr=9.05e-04
OK    epoch 2/5  train_loss=0.1098  train_acc=96.74%  lr=6.55e-04
OK    epoch 3/5  train_loss=0.0743  train_acc=97.78%  lr=3.45e-04
OK    epoch 4/5  train_loss=0.0530  train_acc=98.39%  lr=9.55e-05
OK    epoch 5/5  train_loss=0.0410  train_acc=98.78%  lr=0.00e+00
INFO  saved weights to model.pt  (run eval.py to test)
```

Eval (numbers measured on the held-out 10k test set - shows generalization):

```
INFO  loading model.pt on cuda
OK    test_acc=97.62%  (9762/10000 correct)
```
