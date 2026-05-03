import torch

from data import loaders
from log import log
from model import MLP

# Where to load the trained weights from. Written by train.py.
CKPT = "model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()  # disable gradient tracking - saves memory + time
def evaluate(model, loader, device):
    # Compute classification accuracy on the given loader. Lives here so
    # train.py and any other consumer share one definition.
    model.eval()  # turn off dropout/batchnorm-train behaviour
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)            # predicted class = argmax logit
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct, total


def main():
    # MNIST ships with a fixed 60k/10k train/test split - the test set
    # never overlaps training. We only touch the test loader here.
    _, test_loader = loaders()
    log("info", f"loading {CKPT} on {DEVICE}")

    # Rebuild the same architecture, then load the saved weights into it.
    model = MLP().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

    # Run the held-out test set and report:
    #   test_acc - fraction of unseen samples predicted correctly
    #              (this is the honest measure of generalization)
    correct, total = evaluate(model, test_loader, DEVICE)
    acc = correct / total
    log("ok", f"test_acc={acc*100:.2f}%  ({correct}/{total} correct)")


if __name__ == "__main__":
    main()
