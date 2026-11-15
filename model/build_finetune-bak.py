from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from utils.simple_tokenizer import SimpleTokenizer  # 导入分词器


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        # 初始化分词器
        self.tokenizer = SimpleTokenizer()  # 添加这行

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        # self.logit_scale = torch.ones([]) * (1 / args.temperature)

        # 将温度系数改为可学习参数，更适合多文本场景
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / args.temperature)))

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        # 生成器（轻量化 Transformer 解码器）实现
        if 'nac' in self.current_task:
            # 轻量化Transformer解码器作为生成器
            self.generator = Transformer(
                width=self.embed_dim,
                layers=self.args.generator_depth,  # 较少的层数，如2-3层
                heads=self.embed_dim // 64
            )
            self.generator_proj = nn.Linear(self.embed_dim, self.args.vocab_size)

            # 初始化生成器权重
            scale = self.generator.width ** -0.5
            proj_std = scale * ((2 * self.generator.layers) ** -0.5)
            for block in self.generator.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=scale)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=(2 * self.generator.width) ** -0.5)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            nn.init.normal_(self.generator_proj.weight, std=proj_std)

        if 'mlm' in args.loss_names or 'nac' in self.current_task:

            # 新增：文本自注意力（强化文本内部关联）
            self.text_self_attn = nn.MultiheadAttention(
                self.embed_dim,
                self.embed_dim // 64,
                batch_first=True
            )
            # 新增：图像自注意力（强化图像内部关联）
            self.image_self_attn = nn.MultiheadAttention(
                self.embed_dim,
                self.embed_dim // 64,
                batch_first=True
            )

            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                             64)
            scale = self.cross_modal_transformer.width ** -0.5

            # 新增：自注意力的LayerNorm
            self.ln_pre_text_self = LayerNorm(self.embed_dim)  # 文本自注意力前的归一化
            self.ln_pre_image_self = LayerNorm(self.embed_dim)  # 图像自注意力前的归一化

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

            # 初始化自注意力层
            nn.init.normal_(self.text_self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.text_self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(self.image_self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.image_self_attn.out_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                             ('gelu', QuickGELU()),
                             ('ln', LayerNorm(self.embed_dim)),
                             ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            # nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            # nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
            nn.init.xavier_normal_(self.mlm_head.dense.weight)
            nn.init.xavier_normal_(self.mlm_head.fc.weight)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):

        # 第一阶段：单模态自注意力增强
        # 文本自注意力（q是文本特征，强化文本内部关联）
        text_self_attn_output = self.text_self_attn(
            self.ln_pre_text_self(q),  # 文本特征归一化
            self.ln_pre_text_self(q),
            self.ln_pre_text_self(q),
            need_weights=False
        )[0]
        # 图像自注意力（k/v是图像特征，强化图像内部关联）
        image_self_attn_output = self.image_self_attn(
            self.ln_pre_image_self(k),  # 图像特征归一化
            self.ln_pre_image_self(k),
            self.ln_pre_image_self(v),
            need_weights=False
        )[0]

        # 第二阶段：跨模态交叉注意力融合（复用原有逻辑，基于增强后的特征）
        cross_attn_output = self.cross_attn(
            self.ln_pre_t(text_self_attn_output),  # 用增强后的文本作为query
            self.ln_pre_i(image_self_attn_output),  # 用增强后的图像作为key
            self.ln_pre_i(image_self_attn_output),  # 用增强后的图像作为value
            need_weights=False
        )[0]

        # 跨模态Transformer处理（原有）
        x = cross_attn_output.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x, modal='cross')
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)

        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    # 噪声检测模块实现
    def detect_noise(self, image_feats, text_feats, caption_ids):
        # 提取中间层特征 F_v (图像Patch特征) 和 F_t (文本Token特征)
        # 这里假设image_feats和text_feats是中间层特征
        F_v = image_feats  # 假设形状为 [batch_size, num_patches, embed_dim]
        F_t = text_feats  # 假设形状为 [batch_size, text_len, embed_dim]

        # 计算相似度矩阵
        similarity = torch.matmul(F_t, F_v.transpose(1, 2))  # [batch_size, text_len, num_patches]
        # 计算每个token的噪声分数
        max_similarity, _ = torch.max(similarity, dim=2)  # [batch_size, text_len]
        noise_scores = 1 - max_similarity  # [batch_size, text_len]

        # 确定阈值 (可通过验证集调整或设为超参数)
        tau = self.args.noise_threshold
        noise_mask = noise_scores > tau  # [batch_size, text_len]

        # 忽略特殊token (<|startoftext|>, <|endoftext|>, 0等)
        special_tokens = [self.tokenizer.encoder["<|startoftext|>"],
                          self.tokenizer.encoder["<|endoftext|>"],
                          self.tokenizer.encoder["<|mask|>"],
                          0]
        for token in special_tokens:
            noise_mask &= (caption_ids != token)

        return noise_mask, noise_scores

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']

        # 从batch中获取数据集预处理好的带噪声文本
        # 假设batch中包含'noisy_tokens'和'original_tokens'键
        noisy_tokens = batch.get('noisy_tokens')
        original_tokens = batch.get('original_tokens')

        # # 获取图像ID用于特征复用（每个图像对应4条文本）
        # image_ids = batch['image_ids']

        image_feats, text_feats = self.base_model(images, caption_ids)

        # --------------------------
        # 1. 复用图像特征（兼容旧PyTorch版本）
        # --------------------------
        # 获取唯一图像ID（兼容各种PyTorch版本）
        # unique_ids = torch.unique(image_ids, sorted=True)
        #
        # # 手动计算inverse_indices（找到每个元素在unique_ids中的位置）
        # inverse_indices = torch.zeros_like(image_ids, device=image_ids.device, dtype=torch.long)
        # for i, id in enumerate(unique_ids):
        #     inverse_indices[image_ids == id] = i
        #
        # # 找到每个唯一ID第一次出现的索引
        # indexes = []
        # for id in unique_ids:
        #     # 找到每个唯一ID第一次出现的位置
        #     indices = torch.where(image_ids == id)[0]
        #     if len(indices) > 0:
        #         indexes.append(indices[0].item())
        # indexes = torch.tensor(indexes, device=image_ids.device)
        #
        # # 只提取唯一图像的特征（避免重复计算）
        # unique_images = images[indexes]
        # unique_image_feats = self.base_model.encode_image(unique_images)
        # unique_i_feats = unique_image_feats[:, 0, :].float()
        # unique_i_feats = F.normalize(unique_i_feats, dim=1)  # 归一化
        #
        # # 根据索引扩展图像特征，与文本数量匹配（4N）
        # i_feats = unique_i_feats[inverse_indices]
        #
        # # --------------------------
        # # 2. 提取文本特征（4条文本独立提取）
        # # --------------------------
        # text_feats = self.base_model.encode_text(caption_ids)
        # t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        # t_feats = F.normalize(t_feats, dim=1)  # 归一化

        i_feats = image_feats[:, 0, :].float()  # 形状为[4N, D]，包含重复图像的特征
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()  # 形状[4N, D]

        # ============ 第二步：【关键】NAC噪声纠正 - 放在最前面！ ============
        # 这样纠正后的t_feats才能被后续所有损失计算使用
        if 'nac' in self.current_task:
            # 1. 噪声检测
            noise_mask, _ = self.detect_noise(image_feats, text_feats, caption_ids)
            # noise_mask: [batch_size, seq_len]

            # 2. 特征归一化
            normalized_text_feats = F.normalize(text_feats, dim=-1)
            normalized_image_feats = F.normalize(image_feats, dim=-1)

            # 3. 生成器计算
            gen_feats = self.cross_former(normalized_text_feats, normalized_image_feats, normalized_image_feats)
            gen_output = self.generator(gen_feats, modal='text')

            # 4. 生成器预测
            gen_logits = self.generator_proj(gen_output).float()
            gen_logits = gen_logits.clamp(-100.0, 100.0)

            # 5. 损失计算（生成器损失）
            mask_token_id = self.tokenizer.encoder.get("<|mask|>", None)
            if mask_token_id is None:
                raise ValueError("Mask token not found in tokenizer")

            mask = (noisy_tokens == mask_token_id)

            if not mask.any():
                print("[WARNING] No masked tokens found in noisy_tokens!")
                ret.update({'gen_loss': torch.tensor(0.0, device=gen_logits.device)})
            else:
                gen_loss = F.cross_entropy(
                    gen_logits[mask].reshape(-1, self.args.vocab_size),
                    original_tokens[mask].reshape(-1).to(gen_logits.device),
                    reduction='mean'
                )
                ret.update({'gen_loss': gen_loss * self.args.gen_loss_weight})

            # 6. 噪声纠正 - 获取纠正后的特征
            if self.training:
                with torch.no_grad():
                    # Step 1: 获取生成器预测
                    gen_pred = torch.argmax(gen_logits, dim=-1)  # [batch_size, seq_len]

                    # Step 2: 构建纠正后的tokens
                    corrected_tokens = caption_ids.clone()

                    # 只在噪声位置替换
                    if noise_mask.any():
                        corrected_tokens[noise_mask] = gen_pred[noise_mask]

                    # Step 3: 重新编码
                    corrected_text_feats = self.base_model.encode_text(corrected_tokens)

                    # Step 4: 提取特征
                    end_pos = corrected_tokens.argmax(dim=-1)  # [batch_size]
                    corrected_t_feats = corrected_text_feats[
                        torch.arange(corrected_text_feats.shape[0], device=corrected_text_feats.device),
                        end_pos
                    ].float()
                    corrected_t_feats = F.normalize(corrected_t_feats, dim=1)

                    # Step 5: 在样本级替换特征
                    # ✅ 关键：这里替换的t_feats会被后续所有损失计算使用
                    has_noise = noise_mask.any(dim=-1)  # [batch_size]
                    t_feats = torch.where(
                        has_noise.unsqueeze(-1),  # [batch_size, 1]
                        corrected_t_feats,  # 纠正后的特征
                        t_feats  # 原始特征
                    )


        # 温度系数
        logit_scale = self.logit_scale.exp()  # 使用exp确保为正数
        # logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        # 计算每个文本与对应图片的相似度（用于评估文本质量），主要应用在ttc方法中
        text_norm = t_feats / t_feats.norm(dim=-1, keepdim=True)
        img_norm = i_feats / i_feats.norm(dim=-1, keepdim=True)
        text_img_sim = (text_norm * img_norm).sum(dim=1)  # [4N]，每个文本与对应图片的相似度
        text_quality = F.softmax(text_img_sim, dim=0)  # 归一化为权重（质量高的文本权重高）

        # --------------------------
        # 3. 计算多文本适配的损失函数
        # --------------------------
        if 'itc' in self.current_task:
            # 跨模态对比损失（适配多文本）
            ret.update({
                'itc_loss': objectives.compute_multi_itc_loss(i_feats, t_feats, logit_scale)
            })

        if 'sdm' in self.current_task:
            ret.update({
                'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)
            })

        if 'cmpm' in self.current_task:
            # 跨模态匹配损失（适配多文本）
            ret.update({
                'cmpm_loss': objectives.compute_multi_cmpm_loss(i_feats, t_feats) * 20
            })

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

        # 添加文本-文本对比损失
        if 'ttc' in self.current_task:
            # 传入文本特征、温度系数和pid（用于区分正例）
            ret.update({'ttc_loss': objectives.compute_ttc(t_feats, logit_scale, batch['pids'], text_quality=text_quality)})

        # 新增：文本一致性损失
        if 'text_consistency' in self.current_task:
            ret.update({
                'text_consistency_loss': objectives.compute_text_consistency_loss(t_feats) * 5
            })

        # 在model/build.py的IRRA类的forward方法中添加
        # 这个代码已经很完美了，不需要修改。现在我先注释，用CLAUDE的试试
        # if 'nac' in self.current_task:
        #     # 1. 噪声检测
        #     noise_mask, _ = self.detect_noise(image_feats, text_feats, caption_ids)
        #
        #     # 2. 特征归一化（保持不变）
        #     normalized_text_feats = F.normalize(text_feats, dim=-1)
        #     normalized_image_feats = F.normalize(image_feats, dim=-1)
        #
        #     # 3. 生成器计算（修改部分）
        #     gen_feats = self.cross_former(normalized_text_feats, normalized_image_feats, normalized_image_feats)
        #     gen_output = self.generator(gen_feats)
        #
        #     # 关键修改：1.转fp32 2.裁剪数值（-100~100，避免exp溢出）
        #     gen_logits = self.generator_proj(gen_output).float()  # 从fp16转fp32
        #     gen_logits = gen_logits.clamp(-100.0, 100.0)  # 裁剪极端值，log_softmax计算稳定
        #
        #     # 4. 损失计算（保持不变）
        #     mask = (noisy_tokens == self.tokenizer.encoder["<|mask|>"])
        #     # 加断言防止mask为空（可选，debug用）
        #     assert mask.any(), "No masked tokens found! Check noisy_tokens or mask token."
        #     gen_loss = F.cross_entropy(
        #         gen_logits[mask].reshape(-1, self.args.vocab_size),
        #         original_tokens[mask].reshape(-1).to(gen_logits.device)
        #     )
        #     ret.update({'gen_loss': gen_loss * self.args.gen_loss_weight})
        #
        #
        #
        #     # # 2. 不再在模型内生成噪声，而是直接使用数据集中的结果
        #     # # 确保数据加载时已经生成了这些字段
        #     # if noisy_tokens is None or original_tokens is None:
        #     #     raise ValueError("batch中未找到带噪声的文本，请检查数据集是否正确生成")
        #     #
        #     # # 3. 使用生成器进行纠正
        #     # gen_feats = self.cross_former(text_feats, image_feats, image_feats)  # 利用现有交叉注意力
        #     # gen_output = self.generator(gen_feats)
        #     # gen_logits = self.generator_proj(gen_output)
        #     #
        #     # # 4. 计算生成器损失（仅对被掩码的位置）
        #     # mask = (noisy_tokens == self.tokenizer.encoder["<|mask|>"])
        #     # gen_loss = F.cross_entropy(
        #     #     gen_logits[mask].reshape(-1, self.args.vocab_size),
        #     #     original_tokens[mask].reshape(-1).to(gen_logits.device)
        #     # )
        #     # ret.update({'gen_loss': gen_loss * self.args.gen_loss_weight})
        #
        #     # 5. 在训练中应用纠正后的文本
        #     if self.training:
        #         # 对检测到的噪声词应用掩码
        #         corrected_tokens = caption_ids.clone()
        #         corrected_tokens[noise_mask] = self.tokenizer.encoder["<|mask|>"]
        #
        #         # 使用生成器预测纠正后的token
        #         with torch.no_grad():
        #             corrected_feats = self.cross_former(
        #                 self.base_model.encode_text(corrected_tokens),
        #                 image_feats,
        #                 image_feats
        #             )
        #             corrected_logits = self.generator_proj(self.generator(corrected_feats))
        #             corrected_pred = torch.argmax(corrected_logits, dim=-1)
        #
        #         # 替换噪声词
        #         corrected_tokens[noise_mask] = corrected_pred[noise_mask]
        #
        #         # 使用纠正后的文本重新计算特征
        #         corrected_text_feats = self.base_model.encode_text(corrected_tokens)
        #         t_feats = corrected_text_feats[
        #             torch.arange(corrected_text_feats.shape[0]), corrected_tokens.argmax(dim=-1)].float()

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({
                'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight
            })

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret

        # 这是原始的，不行的话，调回来即可。
        # image_feats, text_feats = self.base_model(images, caption_ids)
        # i_feats = image_feats[:, 0, :].float()
        # # i_feats = image_feats.float() # for CLIP ResNet visual model
        # t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        #
        # logit_scale = self.logit_scale
        # ret.update({'temperature': 1 / logit_scale})
        #
        # if 'itc' in self.current_task:
        #     ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})
        #
        # if 'sdm' in self.current_task:
        #     ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
        #
        # if 'cmpm' in self.current_task:
        #     ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})
        #
        # if 'id' in self.current_task:
        #     image_logits = self.classifier(i_feats.half()).float()
        #     text_logits = self.classifier(t_feats.half()).float()
        #     ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})
        #
        #     image_pred = torch.argmax(image_logits, dim=1)
        #     text_pred = torch.argmax(text_logits, dim=1)
        #
        #     image_precision = (image_pred == batch['pids']).float().mean()
        #     text_precision = (text_pred == batch['pids']).float().mean()
        #     ret.update({'img_acc': image_precision})
        #     ret.update({'txt_acc': text_precision})
        #
        # if 'mlm' in self.current_task:
        #     mlm_ids = batch['mlm_ids']
        #
        #     mlm_feats = self.base_model.encode_text(mlm_ids)
        #
        #     x = self.cross_former(mlm_feats, image_feats, image_feats)
        #
        #     x = self.mlm_head(x)  # [batch_size, text_len, num_colors]
        #
        #     scores = x.float().reshape(-1, self.args.vocab_size)
        #     mlm_labels = batch['mlm_labels'].reshape(-1)
        #     ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})
        #
        #     pred = scores.max(1)[1]
        #     mlm_label_idx = torch.nonzero(mlm_labels)
        #     acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
        #     ret.update({'mlm_acc': acc})
        #
        # return ret


def build_finetune_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model