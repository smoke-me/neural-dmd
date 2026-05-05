import math

from torch.optim.lr_scheduler import CosineAnnealingLR

# Learning-rate scheduling lives here so train.py stays focused on the
# training loop. Swap implementations (StepLR, OneCycleLR, ReduceLROnPlateau,
# ...) by changing this module alone.
#
# Two interfaces:
#   - cosine(opt, total_steps) ........... PyTorch scheduler used in training.
#   - cosine_lr_at(step, total_steps, lr_max, min_lr) ... pure function. Used
#     by the DMDc forecast: at forecast time the optimizer no longer exists,
#     but DMDc still needs u_k = LR(k) for every step it iterates through.


def cosine(opt, total_steps, min_lr=0.0):
    # Cosine annealing: LR follows the right half of a cosine curve from the
    # optimizer's current LR down to `min_lr` over `total_steps` calls to
    # scheduler.step(). Intuition:
    #   - early steps keep LR high so the optimizer can explore broadly
    #   - late steps drop LR smoothly so updates fine-tune the minimum
    # We step once per *batch* (not per epoch), so total_steps must equal
    # epochs * batches_per_epoch.
    return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)


def cosine_lr_at(step: int, total_steps: int, lr_max: float, min_lr: float = 0.0) -> float:
    # Closed-form LR at any training step, mirroring PyTorch's
    # CosineAnnealingLR formula (T_max = total_steps, eta_min = min_lr).
    #
    # PyTorch's recursion equals this closed form when called once per step
    # starting from step=1 (the first sched.step() advances to step=1):
    #     lr(t) = min_lr + 0.5 * (lr_max - min_lr) * (1 + cos(pi * t / T_max))
    # For t = 0 this gives lr_max; for t = T_max it gives min_lr.
    t = max(0, min(total_steps, step))
    return min_lr + 0.5 * (lr_max - min_lr) * (1.0 + math.cos(math.pi * t / total_steps))


def current_lr(opt):
    # Read the live LR straight off the optimizer. Schedulers mutate the
    # optimizer's param_groups, so this reflects whatever the schedule set.
    return opt.param_groups[0]["lr"]
