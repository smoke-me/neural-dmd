from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Pre-processing applied to each MNIST image:
#   1. ToTensor      - convert PIL image (0-255) to float tensor (0-1)
#   2. Normalize     - shift/scale so pixel values have ~zero mean, unit std.
#                      0.1307 and 0.3081 are the known mean/std of MNIST.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def loaders(batch_size=64, root="./data"):
    # Build train and test DataLoaders.
    # `download=True` fetches MNIST into `root` on first run, then reuses it.
    train = datasets.MNIST(root, train=True,  download=True, transform=transform)
    test  = datasets.MNIST(root, train=False, download=True, transform=transform)
    return (
        # shuffle=True so the model sees batches in random order each epoch.
        DataLoader(train, batch_size=batch_size, shuffle=True),
        # Bigger batch for eval is fine; no gradients tracked there.
        DataLoader(test,  batch_size=1000),
    )
