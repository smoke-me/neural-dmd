import sys
import time
from pathlib import Path

# Make the neural_dmd package importable when this script is launched
# directly (e.g. `python scripts/train.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from neural_dmd import config as C
from neural_dmd.data import get_loaders
from neural_dmd.log import banner, log, progress
from neural_dmd.metrics import LOSSES
from neural_dmd.model import MLP
from neural_dmd.schedule import cosine, current_lr
from neural_dmd.snapshots import Recorder


def train_epoch(model, loader, opt, sched, recorder, loss_fn, *,
                epoch_idx: int, total_epochs: int, base_step: int):
    # Run one full pass over the training set. Returns mean loss + accuracy
    # measured on the training data (NOT a measure of generalization).
    # Logs intra-epoch progress every 25% of batches so the user can see
    # how far through the epoch we are even on slow hardware.
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    t_epoch = time.time()

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x, y = x.to(C.DEVICE), y.to(C.DEVICE)

        opt.zero_grad()                          # clear last step's gradients
        logits = model(x)                        # forward pass, raw scores
        loss = loss_fn(logits, y)                # configurable loss (config.LOSS)
        loss.backward()                          # back-prop gradients
        opt.step()                               # update weights
        sched.step()                             # advance LR schedule

        recorder.step(model, opt)                # capture (x_k, u_k) if due

        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += y.size(0)

        progress(f"epoch {epoch_idx}/{total_epochs} batches",
                 batch_idx, n_batches, t_epoch, every_pct=25.0)

    return total_loss / total, correct / total


def main():
    t0 = time.time()

    # 1. Load the chosen dataset. MNIST/FashionMNIST/CIFAR-10 ship a fixed
    #    train/test split - the test set stays untouched during training.
    banner("training start",
           dataset=C.DATASET, arch=C.ARCH, device=C.DEVICE,
           epochs=C.EPOCHS, batch=C.BATCH_SIZE, lr=C.LR, loss=C.LOSS)

    train_loader, _ = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)

    total_steps = C.EPOCHS * len(train_loader)
    fit_steps   = max(2, int(C.FIT_FRAC * total_steps))

    n_fit_snaps      = (fit_steps + C.SNAP_FIT_EVERY - 1) // C.SNAP_FIT_EVERY
    n_forecast_snaps = (total_steps - fit_steps + C.SNAP_FORECAST_EVERY - 1) // C.SNAP_FORECAST_EVERY

    log("info",
        f"trajectory plan: total_steps={total_steps}  fit_steps={fit_steps}  "
        f"snap_fit_every={C.SNAP_FIT_EVERY} -> ~{n_fit_snaps} fit snaps  "
        f"snap_forecast_every={C.SNAP_FORECAST_EVERY} -> ~{n_forecast_snaps} forecast snaps  "
        f"~{(n_fit_snaps + n_forecast_snaps) * 109386 * 4 / 1e6:.0f}MB on disk")

    # 2. Build model, optimizer, LR scheduler, loss function.
    model   = MLP(C.ARCH).to(C.DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=C.LR)
    sched   = cosine(opt, total_steps, min_lr=C.LR_MIN)
    loss_fn = LOSSES[C.LOSS]

    # 3. Snapshot recorder. Two cadences (dense in fit region, sparse in
    #    forecast region) so DMDc has a lot of fit data but the comparison
    #    plot stays a small set of points to evaluate on.
    recorder = Recorder(
        fit_every=C.SNAP_FIT_EVERY,
        forecast_every=C.SNAP_FORECAST_EVERY,
        fit_steps=fit_steps,
        total_steps=total_steps,
        control_fn=C.control_fn,
    )

    # 4. Train. Each line shows train loss, train accuracy, and current lr.
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

    # 5. Persist artefacts. analyze.py + the plot scripts read these.
    banner("training done",
           total_seconds=f"{time.time() - t0:.1f}",
           snapshots_taken=len(recorder.X))

    torch.save(model.state_dict(), C.CKPT_PATH)
    log("info", f"saved weights to {C.CKPT_PATH}")

    recorder.save(C.SNAP_PATH)
    log("ok",
        f"saved {len(recorder.X)} snapshots to {C.SNAP_PATH}  "
        f"(now run analyze.py then the plot_*.py scripts)")


if __name__ == "__main__":
    main()
