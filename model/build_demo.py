from model import objectives
from utils.iotools import load_train_configs
from clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import random

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,
                                                                      args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:

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
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

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
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

        # 原有跨模态注意力机制
        # x = self.cross_attn(
        #     self.ln_pre_t(q),
        #     self.ln_pre_i(k),
        #     self.ln_pre_i(v),
        #     need_weights=False)[0]
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # x = self.cross_modal_transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        #
        # x = self.ln_post(x)
        # return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss': objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            #ret.update({'sdm_loss': objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})
            # This is use infonce_loss，later to rename it.
            ret.update({'sdm_loss': objectives.compute_infonce(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss': objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update(
                {'id_loss': objectives.compute_id(image_logits, text_logits, batch['pids']) * self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels) * self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model



# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载配置文件
config_file = '../logs/CUHK-PEDES/20250612_203558_baseline/configs.yaml'  # 替换为实际的配置文件路径
args = load_train_configs(config_file)
args.training = False

# 手动设置必要的参数
image_size = 224  # 图像尺寸
batch_size = 4    # 批次大小
seq_length = 77   # 文本序列长度
vocab_size = 49152  # 词汇表大小
#
# 构建模型
model = build_model(args, num_classes=11003)  # CUHK-PEDES的类别数
model.to(device)
#
# # 加载预训练模型
# # checkpointer = Checkpointer(model)
# # checkpointer.load(f=os.path.join(args.output_dir, 'best.pth'))  # 替换为实际的预训练模型路径
#
# 生成随机测试数据
random_images = torch.randn(batch_size, 3, image_size, image_size).to(device)
random_text = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)

# 测试图像输入
img_batch = {
    'images': random_images,
    'caption_ids': None,  # 图像输入时文本可以为None
    'pids': torch.randint(0, 11003, (batch_size,)).to(device),  # 随机生成人物ID
    'mlm_ids': None,  # 图像输入时MLM相关可以为None
    'mlm_labels': None  # 图像输入时MLM相关可以为None
}
img_output = model(img_batch)
print("图像输入输出 shape:", {k: v.shape if hasattr(v, 'shape') else v for k, v in img_output.items()})

# 测试文本输入
text_batch = {
    'images': None,  # 文本输入时图像可以为None
    'caption_ids': torch.zeros(batch_size, seq_length).to(device),  # 假设caption_ids是one-hot编码
    'pids': torch.randint(0, 11003, (batch_size,)).to(device),  # 随机生成人物ID
    'mlm_ids': None,  # 文本输入时MLM相关可以为None
    'mlm_labels': None  # 文本输入时MLM相关可以为None
}
text_output = model(text_batch)
print("文本输入输出 shape:", {k: v.shape if hasattr(v, 'shape') else v for k, v in text_output.items()})

# 测试多模态输入
multimodal_batch = {
    'images': random_images,
    'caption_ids': torch.zeros(batch_size, seq_length).to(device),  # 假设caption_ids是one-hot编码
    'pids': torch.randint(0, 11003, (batch_size,)).to(device),  # 随机生成人物ID
    'mlm_ids': None,  # 测试时可以不使用MLM
    'mlm_labels': None  # 测试时可以不使用MLM
}

outputs = model(random_images, random_text)
print(random_text)
print('hehe')