"""Training-time noise-aware caption correction (stage-0 D01--D03).

Detection and token replacement are discrete. The masked generation CE trains
the fusion/generator and the original backbone pass; retrieval gradients do not
pass through argmax or the corrected caption's stopped-gradient re-encoding.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class NoiseAwareCorrection(nn.Module):
    """Fuse [B,L,D] text and [B,N,D] image sequences and repair ordinary tokens.

    Image position zero is CLS and is excluded from cosine detection. Padding is
    positional: ordinary CLIP token ID zero remains a valid replacement. The
    generator is non-autoregressive, with no causal mask or legacy modal path.
    """

    def __init__(self, embed_dim=512, vocab_size=49408, sot_token_id=49406,
                 eot_token_id=49407, mask_token_id=None, noise_threshold=0.7,
                 cmt_depth=4, generator_depth=6, heads=None, dropout=0.0):
        super().__init__()
        self.embed_dim = _positive_integer(embed_dim, "embed_dim")
        self.vocab_size = _positive_integer(vocab_size, "vocab_size")
        heads = max(1, embed_dim // 64) if heads is None else heads
        heads = _positive_integer(heads, "heads")
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        self.cmt_depth = _positive_integer(cmt_depth, "cmt_depth")
        self.generator_depth = _positive_integer(generator_depth, "generator_depth")
        if not math.isfinite(noise_threshold) or not 0 <= noise_threshold <= 2:
            raise ValueError("noise_threshold must be finite and in [0, 2]")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("nac_dropout must be finite and in [0, 1)")
        self.noise_threshold = float(noise_threshold)
        self.sot_token_id, self.eot_token_id = sot_token_id, eot_token_id
        self.mask_token_id = mask_token_id
        self.control_token_ids = tuple(dict.fromkeys(
            value for value in (sot_token_id, eot_token_id, mask_token_id) if value is not None))
        for value in self.control_token_ids:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < vocab_size:
                raise ValueError("Control token IDs must be inside the vocabulary")
        if sot_token_id == eot_token_id or len(self.control_token_ids) >= vocab_size:
            raise ValueError("NAC requires distinct SOT/EOT and an ordinary token vocabulary")

        self.text_self_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.image_self_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.ln_text = nn.LayerNorm(embed_dim)
        self.ln_image = nn.LayerNorm(embed_dim)
        self.ln_cross_text = nn.LayerNorm(embed_dim)
        self.ln_cross_image = nn.LayerNorm(embed_dim)

        def transformer(depth):
            layer = nn.TransformerEncoderLayer(
                embed_dim, heads, dim_feedforward=embed_dim * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(embed_dim),
                                         enable_nested_tensor=False)

        self.cross_modal_transformer = transformer(self.cmt_depth)
        self.generator = transformer(self.generator_depth)
        self.vocab_projection = nn.Linear(embed_dim, vocab_size)

    def _token_masks(self, token_ids, attention_mask):
        if token_ids.ndim != 2 or token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token_ids must be integer [B,L]")
        if token_ids.numel() and ((token_ids < 0) | (token_ids >= self.vocab_size)).any():
            raise ValueError("token_ids contain values outside the vocabulary")
        ends = token_ids.eq(self.eot_token_id)
        if not ends.sum(1).eq(1).all():
            raise ValueError("Each caption must contain exactly one EOT token")
        positions = ends.long().argmax(1)
        indices = torch.arange(token_ids.shape[1], device=token_ids.device)[None]
        valid = indices <= positions[:, None]
        if attention_mask is not None:
            if attention_mask.shape != token_ids.shape or attention_mask.device != token_ids.device:
                raise ValueError("attention_mask must match token_ids")
            if (ends & ~attention_mask.bool()).any():
                raise ValueError("The EOT token must be valid in attention_mask")
            valid = valid & attention_mask.bool()
        ordinary = valid & (indices > 0) & (indices < positions[:, None])
        for value in self.control_token_ids:
            ordinary = ordinary & token_ids.ne(value)
        return valid, ordinary

    @staticmethod
    def _boolean_mask(mask, token_ids, name):
        if (not torch.is_tensor(mask) or mask.shape != token_ids.shape
                or mask.dtype != torch.bool or mask.device != token_ids.device):
            raise ValueError(f"{name} must be boolean [B,L] on the token_ids device")
        return mask

    def detect_noise(self, image_features, text_features, ordinary_mask):
        """Return the visual mask M and 1-max local-patch cosine similarity."""
        with torch.no_grad(), torch.autocast(device_type=text_features.device.type, enabled=False):
            if image_features.shape[1] <= 1:
                scores = text_features.new_zeros(text_features.shape[:2], dtype=torch.float32)
                return torch.zeros_like(ordinary_mask), scores
            text = F.normalize(text_features.float(), dim=-1, eps=1e-8)
            patches = F.normalize(image_features[:, 1:].float(), dim=-1, eps=1e-8)
            similarity = text @ patches.transpose(1, 2)
            scores = 1 - similarity.max(dim=-1).values.clamp(-1, 1)
            return (scores > self.noise_threshold) & ordinary_mask, scores

    def forward(self, image_features, text_features, token_ids, attention_mask=None,
                clean_ids=None, target_valid_mask=None, corruption_mask=None):
        if (image_features.ndim != 3 or text_features.ndim != 3
                or image_features.shape[0] != token_ids.shape[0]
                or text_features.shape[:2] != token_ids.shape
                or image_features.shape[-1] != self.embed_dim
                or text_features.shape[-1] != self.embed_dim or image_features.shape[1] < 1):
            raise ValueError("NAC expects image [B,N,D] with CLS and text [B,L,D] aligned to token_ids")
        valid, ordinary = self._token_masks(token_ids, attention_mask)
        if clean_ids is not None:
            if (clean_ids.shape != token_ids.shape or clean_ids.device != token_ids.device
                    or clean_ids.dtype not in (torch.int32, torch.int64)):
                raise ValueError("clean_ids must be integer [B,L] aligned to token_ids")
            if clean_ids.numel() and ((clean_ids < 0) | (clean_ids >= self.vocab_size)).any():
                raise ValueError("clean_ids contain values outside the vocabulary")
            if target_valid_mask is None:
                raise ValueError("clean_ids requires an explicit target_valid_mask")
        if target_valid_mask is not None:
            self._boolean_mask(target_valid_mask, token_ids, "target_valid_mask")
            if clean_ids is None and target_valid_mask.any():
                raise ValueError("Valid generation targets require clean_ids")
        if corruption_mask is not None:
            self._boolean_mask(corruption_mask, token_ids, "corruption_mask")
        detected, scores = self.detect_noise(image_features, text_features, ordinary)

        # Zero ignored text positions before attention as well as masking keys.
        text = text_features.to(self.ln_text.weight.dtype).masked_fill(~valid[..., None], 0)
        image = image_features.to(self.ln_image.weight.dtype)
        normalized = self.ln_text(text)
        text = self.text_self_attn(normalized, normalized, normalized,
                                   key_padding_mask=~valid, need_weights=False)[0]
        normalized = self.ln_image(image)
        image = self.image_self_attn(normalized, normalized, normalized, need_weights=False)[0]
        query, memory = self.ln_cross_text(text), self.ln_cross_image(image)
        fused = self.cross_attn(query, memory, memory, need_weights=False)[0]
        fused = self.cross_modal_transformer(fused, src_key_padding_mask=~valid)
        generated = self.generator(fused, src_key_padding_mask=~valid)
        logits = self.vocab_projection(generated)

        loss_mask = torch.zeros_like(detected) if clean_ids is None else detected & target_valid_mask
        count = loss_mask.sum()
        if loss_mask.any():
            gen_loss = F.cross_entropy(logits[loss_mask].float(), clean_ids[loss_mask].long())
        else:
            # Keep the zero connected to the generator and the original encoders.
            gen_loss = logits.float().sum() * 0.0
        with torch.no_grad():
            prediction_logits = logits.detach().clone()
            prediction_logits[..., list(self.control_token_ids)] = -torch.inf
            predictions = prediction_logits.argmax(dim=-1)
            corrected_ids = torch.where(detected, predictions.to(token_ids.dtype), token_ids)
        metrics = {
            "nac_detected_tokens": detected.sum(),
            "nac_detected_samples": detected.any(1).sum(),
            "nac_valid_target_tokens": count,
            "nac_noise_score_mean": scores.masked_fill(~ordinary, 0).sum() / ordinary.sum().clamp_min(1),
        }
        if corruption_mask is not None:
            truth = corruption_mask & ordinary
            metrics.update(nac_corrupted_tokens=truth.sum(),
                           nac_detection_true_positives=(detected & truth).sum(),
                           nac_detection_false_positives=(detected & ~truth).sum(),
                           nac_detection_false_negatives=(~detected & truth).sum())
        return {"logits": logits, "detected_noise_mask": detected, "noise_scores": scores,
                "corrected_ids": corrected_ids, "gen_loss": gen_loss,
                "valid_token_count": count, "metrics": {k: v.detach() for k, v in metrics.items()}}
