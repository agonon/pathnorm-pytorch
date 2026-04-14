import bisect
import torch
import numpy as np

# Global variables
X_LR_DECAY_EPOCH = [30 / 90, 60 / 90, 80 / 90]
X_WARMUP_EPOCH_MULTISTEP = 5 / 90
X_WARMUP_EPOCH_COSINE = 5 / 90
WARMUP_ITERATIONS_COSINE = 10000


class WarmupMultistep:
    """Here, iteration = the number of batch seen at training (not the number of epochs)"""

    def __init__(self, warmup_iterations, milestones_in_iterations, gamma):
        self.warmup_iterations = warmup_iterations
        self.milestones_in_iterations = sorted(milestones_in_iterations)
        self.gamma = gamma
        assert self.milestones_in_iterations[0] > warmup_iterations

    def __call__(self, iteration):
        if iteration <= self.warmup_iterations:
            factor = iteration / self.warmup_iterations
        else:
            power = bisect.bisect_right(
                self.milestones_in_iterations, iteration
            )
            factor = self.gamma**power
        return factor


def scheduler_linear_warmup_and_multistep(
    optimizer, gamma, warmup_iterations, milestones_in_iterations
):
    """This scheduler will be called at each new batch, not at each epoch"""
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupMultistep(warmup_iterations, milestones_in_iterations, gamma),
    )
    return scheduler


class WarmupCosine:
    def __init__(self, warmup_end, max_iter, factor_min):
        self.max_iter = max_iter
        self.warmup_end = warmup_end
        self.factor_min = factor_min

    def __call__(self, iteration):
        if iteration < self.warmup_end:
            factor = iteration / self.warmup_end
        else:
            iteration = iteration - self.warmup_end
            max_iter = self.max_iter - self.warmup_end
            iteration = (iteration / max_iter) * np.pi
            factor = self.factor_min + 0.5 * (1 - self.factor_min) * (
                np.cos(iteration) + 1
            )
        return factor


def scheduler_linear_warmup_and_cosine(
    optimizer, initial_lr, warmup_iterations, max_iterations, min_lr=0.0
):
    # min_lr = 0 because: https://github.com/google-research/big_vision/blob/47ac2fd075fcb66cadc0e39bd959c78a6080070d/big_vision/utils.py#L929
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        WarmupCosine(
            warmup_iterations,
            max_iterations,
            min_lr / initial_lr,
        ),
    )
    return scheduler


def get_scheduler(args, optimizer, len_train_loader):
    total_num_iterations = args.epochs * len_train_loader

    if args.lr_scheduler == "constant":
        print("=> Scheduler = constant lr")
        scheduler = None
    elif args.lr_scheduler == "cosine":
        print("=> Scheduler = cosine")
        warmup_iterations = int(X_WARMUP_EPOCH_COSINE * total_num_iterations)
        scheduler = scheduler_linear_warmup_and_cosine(
            optimizer,
            initial_lr=args.lr,
            warmup_iterations=warmup_iterations,
            max_iterations=total_num_iterations,
        )
        print(f"scheduler warmup iterations: {warmup_iterations}")
        print(f"total_num_iterations: {total_num_iterations}")
    elif args.lr_scheduler == "multi-step":
        print("=> Scheduler = multi-step")
        warmup_iterations = int(
            X_WARMUP_EPOCH_MULTISTEP * total_num_iterations
        )
        milestones_in_iterations = [
            int(x * total_num_iterations) for x in X_LR_DECAY_EPOCH
        ]
        scheduler = scheduler_linear_warmup_and_multistep(
            optimizer,
            gamma=0.1,
            warmup_iterations=warmup_iterations,
            milestones_in_iterations=milestones_in_iterations,
        )
        print(f"scheduler warmup iterations: {warmup_iterations}")
        print(
            f"scheduler milestones in iteration number: {milestones_in_iterations}"
        )
    else:
        raise NotImplementedError
    return scheduler
