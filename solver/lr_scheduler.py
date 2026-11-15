"""Epoch scheduler. Call step() after each completed training epoch."""
from bisect import bisect_right
from math import cos, pi
from torch.optim.lr_scheduler import LRScheduler


class LRSchedulerWithWarmup(LRScheduler):
    def __init__(self, optimizer, milestones=(), gamma=0.1, mode="cosine",
                 warmup_factor=0.1, warmup_epochs=5, warmup_method="linear",
                 total_epochs=30, target_lr=0, power=0.9, last_epoch=-1):
        if list(milestones) != sorted(milestones):
            raise ValueError("Milestones must be sorted")
        if mode not in {"step", "exp", "poly", "cosine", "linear"}:
            raise ValueError(f"Unsupported scheduler mode: {mode}")
        if warmup_method not in {"constant", "linear"}:
            raise ValueError("warmup_method must be constant or linear")
        if not isinstance(total_epochs, int) or total_epochs <= 0:
            raise ValueError("total_epochs must be a positive integer")
        if not isinstance(warmup_epochs, int) or not 0 <= warmup_epochs < total_epochs:
            raise ValueError("Require 0 <= warmup_epochs < total_epochs")
        if not 0 < warmup_factor <= 1 or gamma <= 0 or power <= 0:
            raise ValueError("warmup_factor must be in (0,1]; gamma and power must be positive")
        if target_lr < 0 or any(target_lr > group["lr"] for group in optimizer.param_groups):
            raise ValueError("target_lr must be between zero and every initial group learning rate")
        self.milestones = tuple(milestones)
        self.gamma = gamma
        self.mode = mode
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = max(0, self.last_epoch)
        if epoch < self.warmup_epochs:
            factor = self.warmup_factor
            if self.warmup_method == "linear":
                factor += (1 - self.warmup_factor) * epoch / self.warmup_epochs
            return [lr * factor for lr in self.base_lrs]
        if self.mode == "step":
            return [lr * self.gamma ** bisect_right(self.milestones, epoch) for lr in self.base_lrs]
        progress = min(1.0, (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs))
        if self.mode == "cosine":
            factor = 0.5 * (1 + cos(pi * progress))
        elif self.mode == "linear":
            factor = 1 - progress
        elif self.mode == "poly":
            factor = (1 - progress) ** self.power
        else:
            factor = self.power ** progress
        return [self.target_lr + (lr - self.target_lr) * factor for lr in self.base_lrs]
