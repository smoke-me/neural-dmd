"""
Print the trained network's metric (default: classification accuracy) on
the held-out MNIST test set.
"""

import sys
import time
from pathlib import Path

# Make the neural_dmd package importable when this script is launched
# directly (e.g. `python scripts/test.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from neural_dmd import config as C
from neural_dmd.data import get_loaders
from neural_dmd.log import banner, log
from neural_dmd.metrics import METRICS, metric_at
from neural_dmd.model import MLP
from neural_dmd.params import flatten_params


def main():
    t = time.time()
    banner("test.py start", dataset=C.DATASET, metric=C.METRIC, device=C.DEVICE)

    _, test_loader = get_loaders(C.DATASET, C.BATCH_SIZE, C.EVAL_BATCH, C.DATA_ROOT)
    metric_fn = METRICS[C.METRIC]

    log("info", f"loading {C.CKPT_PATH}")
    net = MLP(C.ARCH).to(C.DEVICE)
    net.load_state_dict(torch.load(C.CKPT_PATH, map_location=C.DEVICE))

    score = metric_at(net, flatten_params(net), test_loader, C.DEVICE, metric_fn)
    log("ok",
        f"trained-model {C.METRIC} = {score*100:.2f}%  "
        f"elapsed={time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
