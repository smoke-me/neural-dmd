import torch
import torch.nn.functional as F

from data import loaders
from log import log
from model import MLP
from schedule import cosine, current_lr

# Hyperparameters - edit these to tweak training.
EPOCHS = 5
LR = 1e-3                          # initial (max) learning rate
CKPT = "model.pt"                  # where trained weights get saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_epoch(model, loader, opt, sched):
    # Run one full pass over the training set and return the average loss
    # and accuracy *measured on the training data itself*. These numbers
    # show how well the model is fitting what it has seen - they are NOT a
    # measure of generalization (use eval.py for that).
    model.train()
    total_loss = 0.0   # sum of per-sample losses
    correct = 0        # count of correctly predicted samples
    total = 0          # total samples seen this epoch

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        opt.zero_grad()                          # clear last step's gradients
        logits = model(x)                        # forward pass, raw scores
        loss = F.cross_entropy(logits, y)        # softmax + NLL in one call
        loss.backward()                          # back-prop gradients
        opt.step()                               # update weights
        sched.step()                             # advance LR schedule one step

        # Accumulate stats. Multiply by batch size so we can divide by the
        # true sample count at the end (last batch may be smaller).
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += y.size(0)

    return total_loss / total, correct / total


def main():
    # 1. Load data. MNIST's built-in train/test split is disjoint, so the
    #    test set stays untouched here - we only train on the train split.
    train_loader, _ = loaders()

    # 2. Build model + optimizer + LR scheduler.
    model = MLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Cosine schedule needs to know the total number of optimizer steps
    # across the whole run. We step per batch, so total = epochs * batches.
    total_steps = EPOCHS * len(train_loader)
    sched = cosine(opt, total_steps)

    log("info", f"device={DEVICE}  epochs={EPOCHS}  lr={LR}  schedule=cosine")

    # 3. Train. Each line shows progress on the *training* data:
    #      train_loss - average cross-entropy loss this epoch
    #      train_acc  - fraction of training samples predicted correctly
    #      lr         - learning rate at the end of the epoch (decaying)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, opt, sched)
        log(
            "ok",
            f"epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"train_acc={train_acc*100:.2f}%  "
            f"lr={current_lr(opt):.2e}",
        )

    # 4. Save weights so eval.py can load and test on the held-out set.
    torch.save(model.state_dict(), CKPT)
    log("info", f"saved weights to {CKPT}  (run eval.py to test)")


if __name__ == "__main__":
    main()
