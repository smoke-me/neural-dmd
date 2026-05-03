import torch.nn as nn


# Plain feed-forward network (multi-layer perceptron).
# Input  : 28x28 MNIST image, flattened to 784 values
# Hidden : 128 then 64 neurons, both with ReLU activation
# Output : 10 logits, one per digit class (0-9)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequential chains layers in order. Each input flows top to bottom.
        self.net = nn.Sequential(
            nn.Flatten(),          # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(784, 128),   # fully connected layer
            nn.ReLU(),             # non-linearity
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),     # raw logits; softmax handled in the loss
        )

    def forward(self, x):
        # Called automatically when you do `model(x)`.
        return self.net(x)
