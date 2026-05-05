"""
Pluggable image-classification dataset loader.

To add a new dataset:
  1. Build a (torchvision Dataset class, transform) pair below.
  2. Register it in DATASETS under any short name.
  3. Set config.DATASET to that name. Update config.ARCH so the first
     layer matches the flattened input dim and the last matches #classes.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard normalisations. Mean/std are dataset-specific.
_mnist_tx = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

_cifar_tx = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616)),
])

# Registry: name -> (Dataset class, transform).
DATASETS = {
    "mnist":         (datasets.MNIST,        _mnist_tx),
    "fashion_mnist": (datasets.FashionMNIST, _mnist_tx),
    "cifar10":       (datasets.CIFAR10,      _cifar_tx),
}


def get_loaders(name: str, batch_size: int, eval_batch: int, root):
    # Look up the Dataset class + transform for the chosen name and build
    # train / test DataLoaders. First call downloads the data into `root`.
    # `root` may be a str or pathlib.Path; torchvision accepts either.
    DS, tx = DATASETS[name]
    root = str(root)
    train = DS(root, train=True,  download=True, transform=tx)
    test  = DS(root, train=False, download=True, transform=tx)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test,  batch_size=eval_batch),
    )
