import numpy as np
import torch

# Helpers to convert a model's parameters to/from a single flat vector x.
# DMDc treats the network's whole parameter set as one state x in R^n.


def flatten_params(model) -> np.ndarray:
    # Concatenate every parameter tensor (weights + biases of every layer)
    # into one 1-D NumPy array of float32. Detached + on CPU so we don't
    # hold autograd references or pin GPU memory.
    return torch.cat(
        [p.detach().reshape(-1) for p in model.parameters()]
    ).cpu().numpy().astype(np.float32)


def load_params(model, vec: np.ndarray) -> None:
    # Inverse of flatten_params: copy values from a flat vector back into
    # the model's parameter tensors in-place. Order must match flatten.
    device = next(model.parameters()).device
    t = torch.as_tensor(vec, dtype=torch.float32, device=device)
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(t[i:i + n].view_as(p))
        i += n
