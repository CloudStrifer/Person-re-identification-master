"""CLIP retrieval with optional training-time NAC, A2DFP and CDFP."""

import math
from collections.abc import Mapping

import torch
from torch import nn

from . import objectives
from .clip_model import CLIP, build_CLIP_from_openai_pretrained
from .a2dfp_module import A2DFP, ContrastiveHardMining
from .nac_module import NoiseAwareCorrection


class NCHNetBaseline(nn.Module):
    def __init__(self, args, num_classes, backbone=None, backbone_cfg=None,
                 load_pretrained=True):
        super().__init__()
        self.args = args
        self.num_classes = int(num_classes)
        self.current_task = [name.strip() for name in
                             getattr(args, "loss_names", "sdm+id").split("+")
                             if name.strip()]
        unsupported = set(self.current_task) - {"sdm", "id"}
        requested_modules = [name for name in ("use_dfp", "MLM", "nam")
                             if getattr(args, name, False)]
        if unsupported or requested_modules:
            raise NotImplementedError(
                "Stage 3 supports SDM/ID with optional --use_nac, --use_a2dfp and --use_cdfp. Disable "
                f"tasks={sorted(unsupported)}, modules={requested_modules}; "
                "control CDFP with --use_cdfp, not a loss_names token. "
                "Enable NAC with --use_nac, not a loss_names token; MLM/legacy objectives are not enabled."
            )
        if not self.current_task or len(set(self.current_task)) != len(self.current_task):
            raise ValueError("loss_names must contain distinct supported tasks, e.g. sdm+id")

        if backbone is not None:
            cfg = backbone_cfg or getattr(backbone, "model_config", None)
            if cfg is None:
                raise ValueError("An injected backbone must provide backbone_cfg or model_config")
            self.base_model = backbone
        elif backbone_cfg is not None:
            cfg = dict(backbone_cfg)
            self.base_model = CLIP(**cfg)
            if load_pretrained:
                raise ValueError("Set load_pretrained=False when reconstructing a saved architecture")
        elif load_pretrained:
            checkpoint = getattr(args, "clip_checkpoint", None)
            self.base_model, cfg = build_CLIP_from_openai_pretrained(
                checkpoint or getattr(args, "pretrain_choice", "ViT-B/16"),
                getattr(args, "img_size", (384, 128)),
                getattr(args, "stride_size", 16),
                download_root=(getattr(args, "download_root", None)
                               or getattr(args, "cache_dir", None)),
                allow_download=getattr(args, "allow_download", False),
            )
        else:
            raise ValueError("Checkpoint reconstruction requires backbone_cfg")

        self.backbone_config = dict(cfg)
        self.embed_dim = int(cfg["embed_dim"])
        self.use_a2dfp = bool(getattr(args, "use_a2dfp", False))
        self.use_cdfp = bool(getattr(args, "use_cdfp", False))
        self.use_nac = bool(getattr(args, "use_nac", False) or getattr(args, "include_nac", False))
        self.eot_token_id = int(getattr(args, "eot_token_id", None)
                                if getattr(args, "eot_token_id", None) is not None
                                else cfg["vocab_size"] - 1)
        if not 0 <= self.eot_token_id < cfg["vocab_size"]:
            raise ValueError("eot_token_id is outside the backbone vocabulary")
        self.sot_token_id = int(getattr(args, "sot_token_id", cfg["vocab_size"] - 2))
        self.mask_token_id = getattr(args, "mask_token_id", None)
        for name, value in (("sot_token_id", self.sot_token_id), ("mask_token_id", self.mask_token_id)):
            if value is not None and (not isinstance(value, int) or not 0 <= value < cfg["vocab_size"]):
                raise ValueError(f"{name} is outside the backbone vocabulary")
        if int(getattr(args, "text_length", cfg["context_length"])) != cfg["context_length"]:
            raise ValueError("text_length must match the CLIP checkpoint context_length")
        self.base_model.float()
        self.freeze_backbone = bool(getattr(args, "freeze_backbone", False))
        self.base_model.requires_grad_(not self.freeze_backbone)

        temperature = float(getattr(args, "temperature", 0.02))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        self.id_loss_weight = float(getattr(args, "id_loss_weight", 1.0))
        self.sdm_loss_weight = float(getattr(args, "sdm_loss_weight", 1.0))
        self.label_smoothing = float(getattr(args, "label_smoothing", 0.1))
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        for name, value in (("id_loss_weight", self.id_loss_weight),
                            ("sdm_loss_weight", self.sdm_loss_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")
        if "id" in self.current_task:
            if self.num_classes < 1:
                raise ValueError("ID loss requires at least one training identity")
            self.classifier = nn.Linear(self.embed_dim, self.num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)

        # Preserve the baseline's state keys and initialization when disabled.
        self.a2dfp_blend_weight = float(getattr(args, "a2dfp_blend_weight", 0.5))
        if self.use_a2dfp:
            if not math.isfinite(self.a2dfp_blend_weight) or not 0 <= self.a2dfp_blend_weight <= 1:
                raise ValueError("a2dfp_blend_weight must be finite and in [0, 1]")
            hidden_dim = getattr(args, "a2dfp_embed_dim", 2048)
            if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
                raise ValueError("a2dfp_embed_dim must be a positive integer hidden dimension")
            self.visual_enhance = A2DFP(input_dim=self.embed_dim, hidden_dim=hidden_dim)
            self.textual_enhance = A2DFP(input_dim=self.embed_dim, hidden_dim=hidden_dim)

        self.cdfp_loss_weight = float(getattr(args, "cdfp_loss_weight", 0.1))
        if self.use_cdfp:
            if not math.isfinite(self.cdfp_loss_weight) or self.cdfp_loss_weight < 0:
                raise ValueError("cdfp_loss_weight must be non-negative and finite")
            if getattr(args, "cdfp_temperature", None) is not None:
                raise ValueError("CDFP uses raw cosine hinge loss; cdfp_temperature is not applicable. Omit this option.")
            self.contrastive_mining = ContrastiveHardMining(
                margin=getattr(args, "cdfp_margin", 0.1),
                k_negatives=getattr(args, "cdfp_k_negatives", 2))

        self.gen_loss_weight = float(getattr(args, "gen_loss_weight", 0.5))
        if self.use_nac:
            if not math.isfinite(self.gen_loss_weight) or self.gen_loss_weight < 0:
                raise ValueError("gen_loss_weight must be non-negative and finite")
            self.nac = NoiseAwareCorrection(
                embed_dim=self.embed_dim, vocab_size=cfg["vocab_size"],
                sot_token_id=self.sot_token_id, eot_token_id=self.eot_token_id,
                mask_token_id=self.mask_token_id,
                noise_threshold=getattr(args, "noise_threshold", getattr(args, "nac_threshold", 0.7)),
                cmt_depth=getattr(args, "cmt_depth", getattr(args, "nac_cmt_depth", 4)),
                generator_depth=getattr(args, "generator_depth", getattr(args, "nac_generator_depth", 6)),
                heads=getattr(args, "nac_heads", None), dropout=getattr(args, "nac_dropout", 0.0))

    def encode_image_sequence(self, images):
        """Return CLS and local patches for later feature-learning stages."""
        return self.base_model.encode_image(images)

    def encode_text_sequence(self, token_ids, attention_mask=None):
        self._eot_positions(token_ids, attention_mask)
        return self.base_model.encode_text(token_ids)

    def encode_image(self, images):
        sequence = self.encode_image_sequence(images)
        return self._image_features(sequence)

    def _image_features(self, sequence):
        if sequence.ndim != 3:
            raise ValueError("The baseline requires a ViT sequence backbone")
        original = sequence[:, 0].float()
        if not self.use_a2dfp or sequence.shape[1] == 1:
            return original
        enhanced = self.visual_enhance(sequence[:, 1:].float())
        alpha = self.a2dfp_blend_weight
        return (1 - alpha) * original + alpha * enhanced

    def _eot_positions(self, token_ids, attention_mask=None):
        if token_ids.ndim != 2 or token_ids.shape[1] != self.backbone_config["context_length"]:
            raise ValueError("token_ids must have shape [B, context_length]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must be integer token IDs")
        if token_ids.numel() and (token_ids.min() < 0 or
                                  token_ids.max() >= self.backbone_config["vocab_size"]):
            raise ValueError("token_ids contain values outside the CLIP vocabulary")
        ends = token_ids.eq(self.eot_token_id)
        if attention_mask is not None:
            if attention_mask.shape != token_ids.shape:
                raise ValueError("attention_mask must have the same shape as token_ids")
            if (ends & ~attention_mask.bool()).any():
                raise ValueError("The EOT token must be valid in attention_mask")
        if not ends.sum(dim=1).eq(1).all():
            raise ValueError("Each caption must contain exactly one EOT token")
        return ends.to(torch.long).argmax(dim=1)

    def encode_text(self, token_ids, attention_mask=None):
        positions = self._eot_positions(token_ids, attention_mask)
        sequence = self.base_model.encode_text(token_ids)
        return self._text_features(sequence, token_ids, positions, attention_mask)

    def _text_local_mask(self, token_ids, positions, attention_mask=None):
        indices = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        # Position-based validity preserves ordinary CLIP token 0 inside a caption.
        local_mask = (indices > 0) & (indices < positions.unsqueeze(1))
        local_mask = local_mask & token_ids.ne(self.sot_token_id) & token_ids.ne(self.eot_token_id)
        if self.mask_token_id is not None:
            local_mask = local_mask & token_ids.ne(self.mask_token_id)
        if attention_mask is not None:
            local_mask = local_mask & attention_mask.bool()
        return local_mask

    def _text_features(self, sequence, token_ids, positions, attention_mask=None):
        original = sequence[torch.arange(len(token_ids), device=token_ids.device), positions].float()
        if not self.use_a2dfp:
            return original
        local_mask = self._text_local_mask(token_ids, positions, attention_mask)
        enhanced = self.textual_enhance(sequence.float(), local_mask)
        alpha = self.a2dfp_blend_weight
        blended = (1 - alpha) * original + alpha * enhanced
        return torch.where(local_mask.any(dim=1, keepdim=True), blended, original)

    def forward(self, batch):
        if not isinstance(batch, Mapping):
            raise TypeError("forward expects a batch mapping with images, caption_ids and pids")
        images, token_ids, pids = batch["images"], batch["caption_ids"], batch["pids"]
        if images.shape[0] != token_ids.shape[0] or pids.ndim != 1 or len(pids) != len(images):
            raise ValueError("Image, text and identity batch sizes must agree")
        if pids.dtype != torch.long:
            raise TypeError("pids must be torch.long continuous training labels")
        if "id" in self.current_task and ((pids < 0) | (pids >= self.num_classes)).any():
            raise ValueError("pids must be in [0, num_classes)")
        correction = None
        if self.use_nac and self.training:
            attention_mask = batch.get("attention_mask")
            positions = self._eot_positions(token_ids, attention_mask)
            image_sequence = self.encode_image_sequence(images)
            text_sequence = self.base_model.encode_text(token_ids)
            correction = self.nac(
                image_sequence, text_sequence, token_ids, attention_mask,
                clean_ids=batch.get("clean_ids"), target_valid_mask=batch.get("target_valid_mask"),
                corruption_mask=batch.get("corruption_mask"))
            corrected_ids = correction["corrected_ids"]
            repair_samples = correction["detected_noise_mask"].any(1)
            indices = repair_samples.nonzero(as_tuple=True)[0]
            if indices.numel():
                # D01: share the current encoder; stop only this re-encoding.
                # Unrepaired samples retain their original differentiable pass.
                with torch.no_grad():
                    repaired = self.base_model.encode_text(corrected_ids[indices])
                text_sequence = text_sequence.index_copy(0, indices, repaired)
            image_features = self._image_features(image_sequence)
            text_features = self._text_features(text_sequence, corrected_ids, positions, attention_mask)
        else:
            image_features = self.encode_image(images)
            text_features = self.encode_text(token_ids, batch.get("attention_mask"))
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        losses = {}
        metrics = {"temperature": scale.detach().reciprocal()}
        total = image_features.sum() * 0.0
        if correction is not None:
            losses["gen"] = correction["gen_loss"]
            total = total + self.gen_loss_weight * losses["gen"]
            metrics.update(correction["metrics"])
        if "sdm" in self.current_task:
            losses["sdm"] = objectives.compute_sdm(image_features, text_features, pids, scale)
            total = total + self.sdm_loss_weight * losses["sdm"]
        if "id" in self.current_task:
            image_logits = self.classifier(image_features).float()
            text_logits = self.classifier(text_features).float()
            losses["id"] = objectives.compute_id(
                image_logits, text_logits, pids, label_smoothing=self.label_smoothing)
            total = total + self.id_loss_weight * losses["id"]
            metrics["img_acc"] = image_logits.detach().argmax(dim=1).eq(pids).float().mean()
            metrics["txt_acc"] = text_logits.detach().argmax(dim=1).eq(pids).float().mean()
        if self.use_cdfp:
            image_loss, image_stats = self.contrastive_mining(image_features, pids)
            text_loss, text_stats = self.contrastive_mining(text_features, pids)
            losses["cdfp"] = (image_loss + text_loss) / 2
            total = total + self.cdfp_loss_weight * losses["cdfp"]
            metrics.update(cdfp_image=image_loss.detach(), cdfp_text=text_loss.detach(),
                cdfp_valid_anchors=image_stats["valid_anchors"],
                cdfp_invalid_anchors=image_stats["invalid_anchors"],
                cdfp_active_anchors_image=(image_stats["anchor_losses"] > 0).sum(),
                cdfp_active_anchors_text=(text_stats["anchor_losses"] > 0).sum())
        result = {"loss": total, "losses": losses, "metrics": metrics}
        if correction is not None:
            result["nac"] = {name: correction[name].detach() for name in
                             ("detected_noise_mask", "noise_scores", "corrected_ids")}
        return result


# Preserve the import name used by old scripts while official entrypoints use
# one baseline implementation.
IRRA = NCHNetBaseline


def build_finetune_model(args, num_classes=11003, backbone=None, backbone_cfg=None,
                         load_pretrained=True):
    return NCHNetBaseline(args, num_classes, backbone, backbone_cfg, load_pretrained)
