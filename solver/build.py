"""Optimizer groups follow the paper's backbone/new-module learning rates."""
import torch
from .lr_scheduler import LRSchedulerWithWarmup


def build_optimizer(args, model):
    base_lr = float(args.lr)
    head_lr = getattr(args, "head_lr", None)
    head_lr = float(head_lr if head_lr is not None else base_lr * getattr(args, "lr_factor", 10.0))
    if base_lr <= 0 or head_lr <= 0:
        raise ValueError("Optimizer learning rates must be positive")
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        while name.startswith("module."):
            name = name[len("module."):]
        backbone = name.startswith(("base_model.", "backbone."))
        group_name, lr = ("backbone", base_lr) if backbone else ("heads", head_lr)
        decay = args.weight_decay
        bias_decay = getattr(args, "weight_decay_bias", None)
        if name.endswith(".bias") and bias_decay is not None:
            decay = bias_decay
        groups.setdefault((group_name, lr, decay), []).append(parameter)
    if not groups:
        raise ValueError("The model has no trainable parameters")
    params = [{"params": values, "lr": lr, "weight_decay": decay, "group_name": name}
              for (name, lr, decay), values in groups.items()]
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=base_lr, momentum=args.momentum)
    kwargs = dict(lr=base_lr, betas=(args.alpha, args.beta), eps=getattr(args, "adam_eps", 1e-8))
    if optimizer_name == "adam":
        return torch.optim.Adam(params, **kwargs)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_lr_scheduler(args, optimizer):
    return LRSchedulerWithWarmup(
        optimizer, milestones=args.milestones, gamma=args.gamma,
        warmup_factor=args.warmup_factor, warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method, total_epochs=args.num_epoch,
        mode=args.lrscheduler, target_lr=args.target_lr, power=args.power)
