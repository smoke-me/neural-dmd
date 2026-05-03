from torch.optim.lr_scheduler import CosineAnnealingLR

# Learning-rate scheduling lives here so train.py stays focused on the
# training loop. Swap implementations (StepLR, OneCycleLR, ReduceLROnPlateau,
# ...) by changing this module alone.


def cosine(opt, total_steps, min_lr=0.0):
    # Cosine annealing: LR follows the right half of a cosine curve from the
    # optimizer's current LR down to `min_lr` over `total_steps` calls to
    # scheduler.step(). Intuition:
    #   - early steps keep LR high so the optimizer can explore broadly
    #   - late steps drop LR smoothly so updates fine-tune the minimum
    # We step once per *batch* (not per epoch), so total_steps must equal
    # epochs * batches_per_epoch.
    return CosineAnnealingLR(opt, T_max=total_steps, eta_min=min_lr)


def current_lr(opt):
    # Read the live LR straight off the optimizer. Schedulers mutate the
    # optimizer's param_groups, so this reflects whatever the schedule set.
    return opt.param_groups[0]["lr"]
