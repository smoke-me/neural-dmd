import torch.nn as nn


class MLP(nn.Module):
    # Feed-forward network with arbitrary layer widths.
    #
    # `sizes` is a list of layer sizes, e.g. [784, 128, 64, 10] gives
    # 784 -> 128 -> 64 -> 10 with ReLU between hidden layers and raw
    # logits at the output. Adjust `sizes[0]` to match the flattened
    # input dimension and `sizes[-1]` to match the number of classes
    # whenever you change dataset.
    def __init__(self, sizes):
        super().__init__()
        layers = [nn.Flatten()]
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:           # no activation after last layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
