"""
Train the model + record the parameter trajectory and control sequence.

Outputs land in the content-addressed data cache:

    outputs/data/<data_hash>/snapshots.npz
    outputs/data/<data_hash>/model.pt
    outputs/data/<data_hash>/train_config.json
    outputs/data/<data_hash>/train.log

If the cache already has snapshots.npz + model.pt for the current
training-relevant config (data_hash), this script logs "cache hit" and
exits in a few hundred ms. Use --force to retrain regardless.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the neural_dmd package importable when launched directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from neural_dmd import config as C
from neural_dmd import experiments as E
from neural_dmd.data import get_loaders
from neural_dmd.log import banner, log, progress
from neural_dmd.metrics import LOSSES
from neural_dmd.model import MLP
from neural_dmd.schedule import cosine, current_lr
from neural_dmd.snapshots import Recorder, inject_snapshot_noise


def train_epoch(model, loader, opt, sched, recorder, loss_fn, *,
                epoch_idx: int, total_epochs: int, base_step: int):
    # Run one full pass over the training set. Returns mean loss + accuracy
    # measured on the training data (NOT a measure of generalisation).
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    t_epoch = time.time()

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)

        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        sched.step()

        recorder.step(model, opt)

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += y.size(0)

        progress(f"epoch {epoch_idx}/{total_epochs} batches",
                 batch_idx, n_batches, t_epoch, every_pct=25.0)

    return total_loss / total, correct / total


def _resolve_fit_window(total_steps: int) -> tuple[int, int]:
    """Compute (fit_start_step, fit_end_step) from config.

    FIT_RANGE wins if set (must be a 2-tuple of ints satisfying
    0 <= start < end <= total_steps). Else fall back to FIT_FRAC.
    Both values are participants in the data_hash, so changing them
    invalidates the training cache and forces a retrain."""
    fr = getattr(C, "FIT_RANGE", None)
    if fr is not None:
        start, end = int(fr[0]), int(fr[1])
        if not (0 <= start < end <= total_steps):
            raise ValueError(
                f"FIT_RANGE={fr} invalid for total_steps={total_steps}")
        return start, end
    end = max(2, int(C.FIT_FRAC * total_steps))
    return 0, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="retrain even if cache hit")
    args = parser.parse_args()

    t0 = time.time()
    dh = E.data_hash()

    # ----- cache check -----
    if E.data_cached() and not args.force:
        log("ok",
            f"train.py: cache HIT for data_hash={dh}  "
            f"({E.data_dir()})  skipping train")
        log("info",
            f"  snapshots: {E.snapshots_path()}")
        log("info",
            f"  model:     {E.model_path()}")
        return

    if E.data_cached() and args.force:
        log("warn",
            f"train.py: cache hit for data_hash={dh} but --force given, retraining")

    # ----- seed for determinism -----
    seed = int(getattr(C, "SEED", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ----- load dataset -----
    banner("training start",
           dataset=C.DATASET, arch=C.ARCH, device=C.DEVICE,
           epochs=C.EPOCHS, batch=C.BATCH_SIZE, lr=C.LR, loss=C.LOSS,
           seed=seed, data_hash=dh)

    train_loader, _ = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)

    total_steps = C.EPOCHS * len(train_loader)
    fit_start_step, fit_end_step = _resolve_fit_window(total_steps)
    fit_window_len = fit_end_step - fit_start_step

    n_fit_snaps      = (fit_window_len + C.SNAP_FIT_EVERY - 1) // C.SNAP_FIT_EVERY
    n_pre_snaps      = fit_start_step // C.SNAP_FORECAST_EVERY + (1 if fit_start_step > 0 else 0)
    n_post_snaps     = (total_steps - fit_end_step + C.SNAP_FORECAST_EVERY - 1) // C.SNAP_FORECAST_EVERY

    log("info",
        f"trajectory plan: total_steps={total_steps}  "
        f"fit_window=[{fit_start_step}, {fit_end_step})  "
        f"snap_fit_every={C.SNAP_FIT_EVERY} -> ~{n_fit_snaps} fit snaps  "
        f"snap_forecast_every={C.SNAP_FORECAST_EVERY} -> "
        f"~{n_pre_snaps} pre-fit + ~{n_post_snaps} post-fit snaps")

    # ----- build model + optimiser + scheduler -----
    model   = MLP(C.ARCH).to(C.DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=C.LR)
    sched   = cosine(opt, total_steps, min_lr=C.LR_MIN)
    loss_fn = LOSSES[C.LOSS]

    recorder = Recorder(
        fit_every=C.SNAP_FIT_EVERY,
        forecast_every=C.SNAP_FORECAST_EVERY,
        fit_start_step=fit_start_step,
        fit_end_step=fit_end_step,
        total_steps=total_steps,
        control_fn=C.control_fn,
    )

    # ----- train -----
    for epoch in range(1, C.EPOCHS + 1):
        loss_v, acc_v = train_epoch(model, train_loader, opt, sched, recorder, loss_fn,
                                    epoch_idx=epoch, total_epochs=C.EPOCHS,
                                    base_step=(epoch - 1) * len(train_loader))
        log("ok",
            f"epoch {epoch}/{C.EPOCHS}  "
            f"train_loss={loss_v:.4f}  "
            f"train_acc={acc_v*100:.2f}%  "
            f"lr={current_lr(opt):.2e}  "
            f"snapshots_so_far={len(recorder.X)}")

    # ----- persist into the data cache -----
    banner("training done",
           total_seconds=f"{time.time() - t0:.1f}",
           snapshots_taken=len(recorder.X),
           data_dir=str(E.data_dir()))

    torch.save(model.state_dict(), E.model_path())
    log("info", f"saved weights to {E.model_path()}")

    # Optionally inject Gaussian noise into the recorded snapshots
    # before saving. Seeded off SEED + a fixed offset so noise is
    # reproducible and decoupled from the training RNG stream.
    inject_snapshot_noise(recorder.X,
                          sigma_rel=float(getattr(C, "NOISE_SIGMA", 0.0)),
                          seed=seed + 12345)

    recorder.save(E.snapshots_path())
    log("ok",
        f"saved {len(recorder.X)} snapshots to {E.snapshots_path()}")

    # write the exact training config used (for reproducibility)
    E.train_config_path().write_text(
        json.dumps(E.train_config_dict(), indent=2, default=str))
    log("info", f"saved train_config.json to {E.train_config_path()}")


if __name__ == "__main__":
    main()
