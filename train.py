import torch

import config as C
from data import get_loaders
from log import log
from metrics import LOSSES
from model import MLP
from schedule import cosine, current_lr
from snapshots import Recorder


def train_epoch(model, loader, opt, sched, recorder, loss_fn):
    # Run one full pass over the training set. Returns mean loss + accuracy
    # measured on the training data (NOT a measure of generalization).
    # The recorder snapshots (x_k, u_k) every C.SNAP_EVERY steps.
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
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

    return total_loss / total, correct / total


def main():
    # 1. Load the chosen dataset. MNIST/FashionMNIST/CIFAR-10 ship a fixed
    #    train/test split - the test set stays untouched during training.
    train_loader, _ = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)

    # 2. Build model, optimizer, LR scheduler, loss function.
    model   = MLP(C.ARCH).to(C.DEVICE)
    opt     = torch.optim.Adam(model.parameters(), lr=C.LR)
    sched   = cosine(opt, C.EPOCHS * len(train_loader))
    loss_fn = LOSSES[C.LOSS]

    # 3. Snapshot recorder. The control variable u_k is whatever
    #    config.control_fn returns - default is the scalar learning rate.
    recorder = Recorder(every=C.SNAP_EVERY, control_fn=C.control_fn)

    log("info",
        f"dataset={C.DATASET}  arch={C.ARCH}  device={C.DEVICE}  "
        f"epochs={C.EPOCHS}  lr={C.LR}  loss={C.LOSS}  snap_every={C.SNAP_EVERY}")

    # 4. Train. Each line shows train loss, train accuracy, and current lr.
    for epoch in range(1, C.EPOCHS + 1):
        loss_v, acc_v = train_epoch(model, train_loader, opt, sched, recorder, loss_fn)
        log("ok",
            f"epoch {epoch}/{C.EPOCHS}  "
            f"train_loss={loss_v:.4f}  "
            f"train_acc={acc_v*100:.2f}%  "
            f"lr={current_lr(opt):.2e}")

    # 5. Persist artefacts. eval.py + plot.py read these.
    torch.save(model.state_dict(), C.CKPT_PATH)
    log("info", f"saved weights to {C.CKPT_PATH}")

    recorder.save(C.SNAP_PATH)
    log("info",
        f"saved {len(recorder.X)} snapshots to {C.SNAP_PATH}  "
        f"(run plot.py and eval.py for DMDc analysis)")


if __name__ == "__main__":
    main()
