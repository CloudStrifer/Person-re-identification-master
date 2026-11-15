import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# 忽略pynvml弃用警告（不影响功能，仅优化输出）
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")


class GatedMultiHeadAttention(nn.Module):
    def __init__(self, d_model=2048, num_heads=16, d_k=128):
        super().__init__()
        self.d_model = d_model  # 模型隐藏层维度（论文1.7B模型用2048）
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_k  # 每个头的维度（论文统一为128）

        # 1. QKV投影层（论文Eq.1）
        self.w_q = nn.Linear(d_model, num_heads * d_k)
        self.w_k = nn.Linear(d_model, num_heads * d_k)
        self.w_v = nn.Linear(d_model, num_heads * d_k)

        # 2. 门控模块（核心：头特异性参数，论文2.2节）
        self.gate_weights = nn.Parameter(torch.randn(num_heads, d_model, d_k))  # 每个头独立参数
        self.gate_bias = nn.Parameter(torch.zeros(num_heads, d_k))  # 每个头的偏置

        # 3. 最终输出投影层（论文Eq.4）
        self.w_o = nn.Linear(num_heads * d_k, d_model)

    def forward(self, x, pre_norm_x):
        """
        x: 原始输入 (batch_size, seq_len, d_model)
        pre_norm_x: pre-norm后的隐藏态（论文指定用于计算门控分数的X）
        返回：最终输出、注意力权重、门控分数、SDPA输出（新增，用于测试打印）
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # -------------------------- 步骤1：QKV投影 + 拆分多头 --------------------------
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # -------------------------- 步骤2：SDPA计算（论文Eq.2） --------------------------
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        sdpa_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, d_k)

        # -------------------------- 步骤3：门控调制（核心，论文Eq.5） --------------------------
        gate_scores = []
        for head in range(self.num_heads):
            head_gate = torch.matmul(pre_norm_x, self.gate_weights[head]) + self.gate_bias[head]
            head_gate = torch.sigmoid(head_gate)  # 稀疏分数（均值~0.116）
            gate_scores.append(head_gate.unsqueeze(1))
        gate_scores = torch.cat(gate_scores, dim=1)
        gated_output = sdpa_output * gate_scores  # 乘法调制

        # -------------------------- 步骤4：拼接多头 + 输出投影 --------------------------
        concat_output = gated_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        final_output = self.w_o(concat_output)

        # 新增返回sdpa_output，供测试打印
        return final_output, attn_weights, gate_scores, sdpa_output


# -------------------------- 测试代码（修复变量作用域问题） --------------------------
if __name__ == "__main__":
    # 模拟输入（贴合论文1.7B-28层模型配置）
    batch_size = 8
    seq_len = 4096
    d_model = 2048
    num_heads = 16
    d_k = 128

    x = torch.randn(batch_size, seq_len, d_model)
    pre_norm_x = F.layer_norm(x, normalized_shape=[d_model])  # 模拟pre-norm步骤

    # 初始化模型并前向传播（获取4个返回值，包含sdpa_output）
    gated_attn = GatedMultiHeadAttention(d_model=d_model, num_heads=num_heads, d_k=d_k)
    output, attn_weights, gate_scores, sdpa_output = gated_attn(x, pre_norm_x)  # 关键修复：接收sdpa_output

    # 打印验证（所有变量均来自返回值，无作用域错误）
    print(f"输入维度：x={x.shape}, pre_norm_x={pre_norm_x.shape}")
    print(f"SDPA输出维度：{sdpa_output.shape}（批次×头数×序列长度×头维度）")
    print(f"门控分数维度：{gate_scores.shape}")
    print(f"门控分数稀疏性（均值）：{gate_scores.mean().item():.3f}（论文最优值~0.116）")
    print(f"最终输出维度：{output.shape}（与输入维度一致）")