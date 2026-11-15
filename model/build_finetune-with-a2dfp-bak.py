from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
from .a2dfp_module import A2DFP_VisualEmbedding, A2DFP_TextualEmbedding
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from utils.simple_tokenizer import SimpleTokenizer


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.tokenizer = SimpleTokenizer()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / args.temperature)))

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'nac' in self.current_task:
            self.generator = Transformer(
                width=self.embed_dim,
                layers=self.args.generator_depth,
                heads=self.embed_dim // 64
            )
            self.generator_proj = nn.Linear(self.embed_dim, self.args.vocab_size)
            scale = self.generator.width ** -0.5
            proj_std = scale * ((2 * self.generator.layers) ** -0.5)
            for block in self.generator.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=scale)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=(2 * self.generator.width) ** -0.5)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            nn.init.normal_(self.generator_proj.weight, std=proj_std)

        if 'mlm' in args.loss_names or 'nac' in self.current_task:
            self.text_self_attn = nn.MultiheadAttention(
                self.embed_dim, self.embed_dim // 64, batch_first=True
            )
            self.image_self_attn = nn.MultiheadAttention(
                self.embed_dim, self.embed_dim // 64, batch_first=True
            )
            self.cross_attn = nn.MultiheadAttention(
                self.embed_dim, self.embed_dim // 64, batch_first=True
            )
            self.cross_modal_transformer = Transformer(
                width=self.embed_dim, layers=args.cmt_depth, heads=self.embed_dim // 64
            )
            scale = self.cross_modal_transformer.width ** -0.5

            self.ln_pre_text_self = LayerNorm(self.embed_dim)
            self.ln_pre_image_self = LayerNorm(self.embed_dim)
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers) ** -0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width) ** -0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.text_self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.text_self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(self.image_self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.image_self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            nn.init.xavier_normal_(self.mlm_head.dense.weight)
            nn.init.xavier_normal_(self.mlm_head.fc.weight)

        # A²DFP modules - Novel Enhancement
        self.use_a2dfp = getattr(args, 'use_a2dfp', True)
        self.a2dfp_loss_weight = getattr(args, 'a2dfp_loss_weight', 0.1)
        self.a2dfp_blend_weight = getattr(args, 'a2dfp_blend_weight', 0.3)

        if self.use_a2dfp:
            self.visual_a2dfp = A2DFP_VisualEmbedding(
                input_dim=self.embed_dim,
                embed_dim=getattr(args, 'a2dfp_embed_dim', 2048),
                num_heads=self.embed_dim // 64
            )
            self.textual_a2dfp = A2DFP_TextualEmbedding(
                input_dim=self.embed_dim,
                embed_dim=getattr(args, 'a2dfp_embed_dim', 2048),
                num_heads=self.embed_dim // 64
            )

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        text_self_attn_output = self.text_self_attn(
            self.ln_pre_text_self(q),
            self.ln_pre_text_self(q),
            self.ln_pre_text_self(q),
            need_weights=False
        )[0]

        image_self_attn_output = self.image_self_attn(
            self.ln_pre_image_self(k),
            self.ln_pre_image_self(k),
            self.ln_pre_image_self(v),
            need_weights=False
        )[0]

        cross_attn_output = self.cross_attn(
            self.ln_pre_t(text_self_attn_output),
            self.ln_pre_i(image_self_attn_output),
            self.ln_pre_i(image_self_attn_output),
            need_weights=False
        )[0]

        x = cross_attn_output.permute(1, 0, 2)
        x = self.cross_modal_transformer(x,  modal='cross')
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)

        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_with_a2dfp(self, image_feats, pids):
        """Enhanced image encoding with A²DFP"""
        try:
            if self.use_a2dfp:
                i_feats_enhanced, losses = self.visual_a2dfp(image_feats, pids)
                return i_feats_enhanced, losses
            return image_feats[:, 0, :].float(), {}
        except Exception as e:
            return image_feats[:, 0, :].float(), {}

    def encode_text_with_a2dfp(self, text_feats, pids):
        """Enhanced text encoding with A²DFP"""
        try:
            if self.use_a2dfp:
                t_feats_enhanced, losses = self.textual_a2dfp(text_feats, pids)
                return t_feats_enhanced, losses
            return text_feats, {}
        except Exception as e:
            return text_feats, {}

    def detect_noise(self, image_feats, text_feats, caption_ids):
        F_v = image_feats
        F_t = text_feats
        similarity = torch.matmul(F_t, F_v.transpose(1, 2))
        max_similarity, _ = torch.max(similarity, dim=2)
        noise_scores = 1 - max_similarity
        tau = self.args.noise_threshold
        noise_mask = noise_scores > tau
        special_tokens = [self.tokenizer.encoder["<|startoftext|>"],
                          self.tokenizer.encoder["<|endoftext|>"],
                          self.tokenizer.encoder["<|mask|>"], 0]
        for token in special_tokens:
            noise_mask &= (caption_ids != token)
        return noise_mask, noise_scores

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        pids = batch['pids']
        noisy_tokens = batch.get('noisy_tokens')
        original_tokens = batch.get('original_tokens')

        image_feats, text_feats = self.base_model(images, caption_ids)

        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        # A²DFP Enhancement (Novel)
        a2dfp_losses = {}
        if self.use_a2dfp:
            try:
                i_feats_a2dfp, i_losses = self.encode_image_with_a2dfp(image_feats, pids)
                t_feats_a2dfp, t_losses = self.encode_text_with_a2dfp(text_feats, pids)

                # Blend with original features
                i_feats = (1 - self.a2dfp_blend_weight) * i_feats + self.a2dfp_blend_weight * i_feats_a2dfp
                t_feats = (1 - self.a2dfp_blend_weight) * t_feats + self.a2dfp_blend_weight * t_feats_a2dfp

                # Accumulate losses
                a2dfp_losses.update(i_losses)
                a2dfp_losses.update(t_losses)
            except Exception as e:
                pass

        logit_scale = self.logit_scale.exp()
        ret.update({'temperature': 1 / logit_scale})

        text_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)
        img_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
        text_img_sim = (text_norm * img_norm).sum(dim=1)
        text_quality = F.softmax(text_img_sim, dim=0)

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_multi_itc_loss(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_multi_cmpm_loss(i_feats, t_feats) * 20})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({
                'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight
            })
            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)
            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'ttc' in self.current_task:
            ret.update(
                {'ttc_loss': objectives.compute_ttc(t_feats, logit_scale, batch['pids'], text_quality=text_quality)})

        if 'text_consistency' in self.current_task:
            ret.update({'text_consistency_loss': objectives.compute_text_consistency_loss(t_feats) * 5})

        if 'nac' in self.current_task:
            noise_mask, _ = self.detect_noise(image_feats, text_feats, caption_ids)
            normalized_text_feats = F.normalize(text_feats, dim=-1)
            normalized_image_feats = F.normalize(image_feats, dim=-1)
            gen_feats = self.cross_former(normalized_text_feats, normalized_image_feats, normalized_image_feats)
            gen_output = self.generator(gen_feats)
            gen_logits = self.generator_proj(gen_output).float()
            gen_logits = gen_logits.clamp(-100.0, 100.0)
            mask_token_id = self.tokenizer.encoder.get("<|mask|>", None)
            if mask_token_id is None:
                raise ValueError("Mask token not found in tokenizer")
            mask = (noisy_tokens == mask_token_id)
            if not mask.any():
                ret.update({'gen_loss': torch.tensor(0.0, device=gen_logits.device)})
            else:
                gen_loss = F.cross_entropy(
                    gen_logits[mask].reshape(-1, self.args.vocab_size),
                    original_tokens[mask].reshape(-1).to(gen_logits.device),
                    reduction='mean'
                )
                ret.update({'gen_loss': gen_loss * self.args.gen_loss_weight})

            if self.training:
                with torch.no_grad():
                    gen_pred = torch.argmax(gen_logits, dim=-1)
                    corrected_tokens = caption_ids.clone()
                    if noise_mask.any():
                        corrected_tokens[noise_mask] = gen_pred[noise_mask]
                    corrected_text_feats = self.base_model.encode_text(corrected_tokens)
                    end_pos = corrected_tokens.argmax(dim=-1)
                    corrected_t_feats = corrected_text_feats[
                        torch.arange(corrected_text_feats.shape[0], device=corrected_text_feats.device),
                        end_pos
                    ].float()
                    corrected_t_feats = F.normalize(corrected_t_feats, dim=1)
                    has_noise = noise_mask.any(dim=-1)
                    t_feats = torch.where(has_noise.unsqueeze(-1), corrected_t_feats, t_feats)

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']
            mlm_feats = self.base_model.encode_text(mlm_ids)
            x = self.cross_former(mlm_feats, image_feats, image_feats)
            x = self.mlm_head(x)
            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({
                'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight
            })
            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            if len(mlm_label_idx) > 0:
                acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
                ret.update({'mlm_acc': acc})

        # Add A²DFP losses with weighting
        for k, v in a2dfp_losses.items():
            if k not in ret:
                ret[k] = v * self.a2dfp_loss_weight

        return ret


def build_finetune_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    convert_weights(model)
    return model