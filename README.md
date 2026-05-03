# neural-dmd

MNIST training: 784 → 128 → 64 → 10 MLP in PyTorch.

## Files

- `model.py` — `MLP` network definition.
- `data.py` — MNIST download, transforms, train/test `DataLoader`s.
- `train.py` — train loop + entry point. Saves weights to `model.pt` at end. Hyperparameters at top.
- `eval.py` — loads `model.pt` and reports test accuracy. Also exports `evaluate()` reused by `train.py`.
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

Train (numbers measured on the training set - shows the model fitting):

```
INFO  device=cuda  epochs=5  lr=0.001
OK    epoch 1/5  train_loss=0.2543  train_acc=92.61%
OK    epoch 2/5  train_loss=0.1098  train_acc=96.74%
...
INFO  saved weights to model.pt  (run eval.py to test)
```

Eval (numbers measured on the held-out 10k test set - shows generalization):

```
INFO  loading model.pt on cuda
OK    test_acc=97.62%  (9762/10000 correct)
```
