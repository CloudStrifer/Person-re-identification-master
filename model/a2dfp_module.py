"""A2DFP local aggregation and CDFP raw-cosine hard mining for NCHNet.

Uses paper equations (16)-(23), (28) and stage-0 decisions D04/D05. The 2048
dimension is an internal projection width. Residual blending belongs to the
caller. CDFP is parameter-free and never rescales similarity by temperature.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _masked_softmax(scores, mask):
    """Normalize valid positions without NaNs for entirely masked rows."""
    work = scores.float() if scores.dtype in (torch.float16, torch.bfloat16) else scores
    valid_rows = mask.any(dim=1, keepdim=True)
    masked = work.masked_fill(~mask, -torch.inf)
    safe = torch.where(valid_rows, masked, torch.zeros_like(masked))
    return F.softmax(safe, dim=1).masked_fill(~mask, 0)


class A2DFP(nn.Module):
    """Independent per-sample local aggregation, [B,N,D] -> [B,D].

    valid_mask is boolean [B,N]. Padding is replaced before projections, so NaN
    padding cannot affect the output. All-invalid samples return graph-connected
    zero; callers must retain their original global features for these samples.
    return_diagnostics=True additionally returns detached attention/pool weights
    and a per-sample valid_samples mask.
    """

    def __init__(self, input_dim=512, hidden_dim=2048):
        super().__init__()
        self.input_dim = _positive_integer(input_dim, "input_dim")
        self.hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        self.attention_fc = nn.Sequential(
            nn.Linear(input_dim, max(1, input_dim // 2)), nn.ReLU(),
            nn.Linear(max(1, input_dim // 2), 1), nn.Sigmoid())
        self.context_gate = nn.Sequential(
            nn.Linear(input_dim, max(1, input_dim // 4)), nn.ReLU(),
            nn.Linear(max(1, input_dim // 4), 1), nn.Sigmoid())
        self.pool_fc = nn.Sequential(
            nn.Linear(input_dim, max(1, input_dim // 2)), nn.ReLU(),
            nn.Linear(max(1, input_dim // 2), 1))
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim))

    def forward(self, features, valid_mask=None, return_diagnostics=False):
        if not torch.is_tensor(features) or features.ndim != 3:
            raise ValueError("features must have shape [B,N,D]")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"features last dimension must be {self.input_dim}")
        if valid_mask is None:
            valid_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        elif (not torch.is_tensor(valid_mask) or valid_mask.shape != features.shape[:2]
              or valid_mask.dtype != torch.bool or valid_mask.device != features.device):
            raise ValueError("valid_mask must be boolean [B,N] on the features device")
        clean = features.masked_fill(~valid_mask.unsqueeze(-1), 0)
        if not torch.isfinite(clean).all():
            raise ValueError("Valid local features must be finite")
        # Match parameter dtype, while allowing autocast to select Linear dtype.
        clean = clean.to(dtype=self.attention_fc[0].weight.dtype)
        importance = self.attention_fc(clean).squeeze(-1)
        context = self.context_gate(clean).squeeze(-1)
        attention = _masked_softmax(importance * context, valid_mask)
        weighted = clean * attention.to(clean.dtype).unsqueeze(-1)
        pooling = _masked_softmax(self.pool_fc(weighted).squeeze(-1), valid_mask)
        pooled = (weighted * pooling.to(weighted.dtype).unsqueeze(-1)).sum(dim=1)
        enhanced = self.projection(pooled)
        valid_samples = valid_mask.any(dim=1)
        enhanced = enhanced.masked_fill(~valid_samples.unsqueeze(-1), 0)
        if return_diagnostics:
            return enhanced, {
                "attention_weights": attention.detach(),
                "pool_weights": pooling.detach(),
                "valid_samples": valid_samples.detach(),
            }
        return enhanced


class ContrastiveHardMining(nn.Module):
    """Hardest-positive / Top-k-negative raw-cosine hinge loss per modality.

    Only anchors with another same-identity sample and at least one negative
    enter the mean. Empty or entirely invalid batches return graph-connected
    zero. Statistics are detached diagnostics, never additional loss terms.
    """

    def __init__(self, margin=0.1, k_negatives=2):
        super().__init__()
        if isinstance(margin, bool) or not isinstance(margin, (int, float)):
            raise ValueError("margin must be a finite nonnegative number")
        if not math.isfinite(margin) or margin < 0:
            raise ValueError("margin must be a finite nonnegative number")
        self.margin = float(margin)
        self.k_negatives = _positive_integer(k_negatives, "k_negatives")

    def forward(self, features, pids):
        if not torch.is_tensor(features) or features.ndim != 2:
            raise ValueError("features must have shape [B,D]")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if features.shape[1] == 0:
            raise ValueError("features must have a nonempty embedding dimension")
        if (not torch.is_tensor(pids) or pids.ndim != 1 or len(pids) != len(features)
                or pids.device != features.device):
            raise ValueError("pids must have shape [B] on the features device")
        if pids.dtype not in (torch.int32, torch.int64):
            raise TypeError("pids must be integer identity labels")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        # Preserve the raw-cosine scale even under CPU/CUDA autocast.
        with torch.autocast(device_type=features.device.type, enabled=False):
            work = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
            count = len(features)
            normalized = F.normalize(work, p=2, dim=-1, eps=1e-8)
            similarities = normalized @ normalized.t()
            same = pids[:, None].eq(pids[None, :])
            diagonal = torch.eye(count, dtype=torch.bool, device=features.device)
            positives, negatives = same & ~diagonal, ~same
            positive_counts, negative_counts = positives.sum(1), negatives.sum(1)
            valid = (positive_counts > 0) & (negative_counts > 0)
            if count:
                positive_values = similarities.masked_fill(~positives, torch.inf).min(dim=1).values
                positive_values = torch.where(positive_counts > 0, positive_values, torch.zeros_like(positive_values))
                selected = similarities.masked_fill(~negatives, -torch.inf).topk(
                    min(self.k_negatives, count), dim=1).values
                selected_valid = torch.isfinite(selected)
                selected_counts = selected_valid.sum(1)
                negative_values = selected.masked_fill(~selected_valid, 0).sum(1) / selected_counts.clamp_min(1)
                anchor_losses = F.relu(negative_values - positive_values + self.margin)
                anchor_losses = torch.where(valid, anchor_losses, torch.zeros_like(anchor_losses))
                loss = anchor_losses.sum() / valid.sum().clamp_min(1)
            else:
                positive_values = work.new_zeros((0,))
                negative_values = work.new_zeros((0,))
                selected_counts = negative_counts.clone()
                anchor_losses = work.new_zeros((0,))
                loss = work.sum() * 0
        statistics = {
            "valid_anchors": valid.sum(), "invalid_anchors": (~valid).sum(),
            "valid_mask": valid,
            "hard_positive": torch.where(valid, positive_values, torch.zeros_like(positive_values)),
            "hard_negative_mean": torch.where(valid, negative_values, torch.zeros_like(negative_values)),
            "anchor_losses": anchor_losses,
            "positive_counts": positive_counts, "negative_counts": negative_counts,
            "selected_negative_counts": selected_counts,
        }
        return loss, {key: value.detach() for key, value in statistics.items()}
