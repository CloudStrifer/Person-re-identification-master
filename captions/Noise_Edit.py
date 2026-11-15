import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class VisualFeatureExtractor(nn.Module):
    """提取图像特征"""

    def __init__(self, visual_encoder_name="google/vit-base-patch16-224"):
        super().__init__()
        self.visual_encoder = AutoModel.from_pretrained(visual_encoder_name)

    def forward(self, pixel_values):
        outputs = self.visual_encoder(pixel_values=pixel_values)
        # 使用[CLS] token作为全局图像特征
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]


class NoiseAwareCorrector(nn.Module):
    """噪声感知纠正器"""

    def __init__(self, hidden_size=768, vocab_size=30522, num_layers=3, nhead=8):
        super().__init__()

        # 轻量级Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=nhead, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 输出层，预测词汇
        self.output_projection = nn.Linear(hidden_size, vocab_size)

        # 用于将图像特征转换为解码器可用的格式
        self.visual_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, text_embeddings, visual_features, text_attention_mask=None):
        """
        text_embeddings: 文本嵌入 [batch_size, seq_len, hidden_size]
        visual_features: 图像特征 [batch_size, hidden_size]
        text_attention_mask: 文本注意力掩码 [batch_size, seq_len]
        """
        # 将图像特征转换为解码器记忆格式
        memory = self.visual_projection(visual_features).unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 通过Transformer解码器
        decoded = self.decoder(
            text_embeddings,
            memory,
            tgt_key_padding_mask=text_attention_mask
        )

        # 预测每个位置的词汇
        logits = self.output_projection(decoded)  # [batch_size, seq_len, vocab_size]
        return logits


class NACModel(nn.Module):
    """完整的NAC模型"""

    def __init__(self, text_encoder_name="bert-base-uncased"):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.visual_extractor = VisualFeatureExtractor()
        self.corrector = NoiseAwareCorrector()

        # 冻结文本编码器和视觉提取器（可选）
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.visual_extractor.parameters():
            param.requires_grad = False

        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.mask_token_id = self.tokenizer.mask_token_id

    def detect_noise(self, input_ids, pixel_values):
        """使用NAM方法检测噪声词"""
        # 获取文本和图像特征
        text_features = self.text_encoder(input_ids=input_ids).last_hidden_state
        visual_features = self.visual_extractor(pixel_values)

        # 计算相似度（简化版）
        # 实际实现应按照原论文计算每个token与所有图像patch的相似度
        text_norm = F.normalize(text_features, p=2, dim=-1)
        visual_norm = F.normalize(visual_features.unsqueeze(1), p=2, dim=-1)
        similarities = torch.sum(text_norm * visual_norm, dim=-1)  # [batch_size, seq_len]

        # 计算噪声分数
        noise_scores = 1 - similarities

        return noise_scores

    def correct_noise(self, input_ids, pixel_values, noise_threshold=0.7):
        """纠正文本中的噪声词"""
        # 检测噪声
        noise_scores = self.detect_noise(input_ids, pixel_values)

        # 创建掩码版本文本
        masked_input_ids = input_ids.clone()

        # 标识需要纠正的位置（噪声分数高于阈值）
        correction_mask = noise_scores > noise_threshold

        # 将噪声词替换为[MASK]
        masked_input_ids[correction_mask] = self.mask_token_id

        # 获取文本嵌入
        text_embeddings = self.text_encoder(input_ids=masked_input_ids).last_hidden_state

        # 获取图像特征
        visual_features = self.visual_extractor(pixel_values)

        # 通过纠正器生成预测
        correction_logits = self.corrector(text_embeddings, visual_features)

        # 获取预测结果（只对掩码位置）
        predicted_ids = torch.argmax(correction_logits, dim=-1)
        corrected_input_ids = input_ids.clone()
        corrected_input_ids[correction_mask] = predicted_ids[correction_mask]

        return corrected_input_ids, correction_mask

    def forward(self, input_ids, pixel_values, attention_mask=None):
        """前向传播（用于训练）"""
        # 获取文本嵌入
        text_embeddings = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        # 获取图像特征
        visual_features = self.visual_extractor(pixel_values)

        # 通过纠正器
        correction_logits = self.corrector(text_embeddings, visual_features, attention_mask)

        return correction_logits


# 训练过程示例
def train_nac(model, dataloader, optimizer, device):
    """训练NAC模型（自监督方式）"""
    model.train()

    for batch in dataloader:
        # 获取数据
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 人为制造掩码：随机选择15%的token进行掩码
        # 其中80%替换为[MASK]，10%随机替换，10%保持不变
        masked_input_ids = input_ids.clone()
        labels = input_ids.clone()

        # 创建随机掩码
        probability_matrix = torch.full(labels.shape, 0.15)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # 只计算被掩码位置的损失

        # 80%的概率替换为[MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        masked_input_ids[indices_replaced] = model.mask_token_id

        # 10%的概率随机替换
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(0, model.tokenizer.vocab_size, labels.shape, device=device)
        masked_input_ids[indices_random] = random_words[indices_random]

        # 剩余10%保持不变

        # 前向传播
        correction_logits = model(masked_input_ids, pixel_values, attention_mask)

        # 计算损失
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            correction_logits.view(-1, model.tokenizer.vocab_size),
            labels.view(-1)
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印损失
        print(f"Loss: {loss.item()}")


# 使用示例
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    model = NACModel().to(device)

    # 创建优化器（只训练纠正器部分）
    optimizer = torch.optim.Adam(model.corrector.parameters(), lr=1e-4)

    # 创建数据加载器（需要实现）
    # dataloader = create_dataloader(...)

    # 训练模型
    # for epoch in range(num_epochs):
    #     train_nac(model, dataloader, optimizer, device)

    # 在ReID训练中使用纠正器
    def reid_training_step(image, text_input_ids):
        # 检测并纠正噪声
        corrected_text, correction_mask = model.correct_noise(text_input_ids, image)

        # 使用纠正后的文本进行ReID训练
        # image_features = reid_model.image_encoder(image)
        # text_features = reid_model.text_encoder(corrected_text)
        # loss = compute_reid_loss(image_features, text_features)

        # return loss


if __name__ == "__main__":
    main()