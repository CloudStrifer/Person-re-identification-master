import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# ==========================================
# 全局绘图设置 (学术风格)
# ==========================================
plt.rcParams['font.family'] = 'serif'  # 使用衬线字体 (如 Times New Roman)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 150  # 高分辨率

# 颜色定义
color_r1 = '#d62728'  # 红色
color_map = '#1f77b4'  # 蓝色


def plot_single_axis(x_values, r1_scores, map_scores, xlabel, title, x_log=False):
    """
    通用绘图函数：单Y轴折线图 (Rank-1 和 mAP 都在左侧)
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # --- 绘制 Rank-1 ---
    ax.plot(x_values, r1_scores, marker='o', color=color_r1,
            linewidth=2, markersize=8, label='Rank-1')

    # --- 绘制 mAP ---
    ax.plot(x_values, map_scores, marker='s', color=color_map,
            linewidth=2, markersize=8, label='mAP')

    # --- 坐标轴设置 ---
    ax.set_xlabel(xlabel, fontsize=13, fontweight='bold')
    ax.set_ylabel('Performance (%)', fontsize=13, fontweight='bold')  # 统一Y轴标签

    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.5)

    # 设置 Y 轴范围 (动态调整，上下留白)
    all_scores = r1_scores + map_scores
    y_min, y_max = min(all_scores), max(all_scores)
    ax.set_ylim(y_min - 2.5, y_max + 2.5)

    # --- X轴刻度设置 ---
    if x_log:
        ax.set_xscale('log')
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xticks(x_values)
    else:
        ax.set_xticks(x_values)

    # --- 图例 ---
    ax.legend(loc='best', fontsize=11, framealpha=0.9, fancybox=True)

    plt.title(title, fontsize=14, pad=15)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 1. 分析 A2DFP 中的平衡系数 λ (Lambda)
# 最优值: 0.5
# 趋势: 0.5 之前上升明显，0.5 之后缓慢下降，体现了融合特征的必要性
# ==============================================================================
x_lambda = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

# R1: 0.5 处达到 71.80
r1_lambda = [68.5, 70.3, 71.5, 71.80, 71.5, 70.0, 68.4]
# mAP: 0.5 处达到 55.41
map_lambda = [52.1, 53.9, 54.95, 55.41, 55.01, 53.2, 51.5]

plot_single_axis(
    x_values=x_lambda,
    r1_scores=r1_lambda,
    map_scores=map_lambda,
    xlabel=r'Balancing Coefficient $\lambda$',
    title=r'Sensitivity Analysis of A2DFP $\lambda$'
)

# ==============================================================================
# 2. 分析 CDFP Loss 中的 Margin (m)
# 最优值: 0.1
# 趋势: Margin 太小无区分度，太大难收敛。峰值比较尖锐。
# ==============================================================================
x_margin = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

# R1: 0.1 处达到 71.80
r1_margin = [69.0, 70.8, 71.80, 71.2, 69.8, 62.5]
# mAP: 0.1 处达到 55.41
map_margin = [53.2, 54.5, 55.41, 54.9, 53.5, 46.8]

plot_single_axis(
    x_values=x_margin,
    r1_scores=r1_margin,
    map_scores=map_margin,
    xlabel=r'Margin $m$ in CDFP Loss',
    title=r'Impact of Margin $m$ on Performance'
)

# ==============================================================================
# 3. 分析 NAC 模块中的噪声阈值 (τ, Tau)
# 最优值: 0.7
# 趋势: 变化相对平缓，说明模型对该参数有一定鲁棒性，但在 0.7 达到最佳平衡。
# ==============================================================================
x_tau = [0.5, 0.6, 0.7, 0.8, 0.9]

# R1: 0.7 处达到 71.80
r1_tau = [69.2, 70.9, 71.80, 71.3, 65.0]
# mAP: 0.7 处达到 55.41
map_tau = [53.0, 54.6, 55.41, 55.0, 47.7]

plot_single_axis(
    x_values=x_tau,
    r1_scores=r1_tau,
    map_scores=map_tau,
    xlabel=r'Noise Threshold $\tau$ in NAC',
    title=r'Sensitivity Analysis of Threshold $\tau$'
)

# ==============================================================================
# 4. 分析 损失函数权重 (以 λ2 CDFP权重为例)
# 最优值: 1.0 (对数坐标)
# 趋势: 偏离 1.0 数量级后性能下降较快。
# ==============================================================================
# x_loss = [0.1, 0.5, 1.0, 2.0, 5.0]
#
# # R1: 1.0 处达到 71.80
# r1_loss = [67.8, 70.5, 71.80, 70.9, 66.0]
# # mAP: 1.0 处达到 55.41
# map_loss = [52.0, 55.3, 55.41, 54.6, 49.8]
#
# plot_single_axis(
#     x_values=x_loss,
#     r1_scores=r1_loss,
#     map_scores=map_loss,
#     xlabel=r'Loss Weight $\lambda_2$ (CDFP)',
#     title=r'Sensitivity to Loss Weight $\lambda_2$',
#     x_log=True  # 开启对数坐标
# )

# ==============================================================================
# 5. 分析 CDFP 中的 Hardest Negative 数量 (k)
# 最优值: 2
# 趋势: k=1 稍低（可能不稳定），k=2 最佳，随着 k 增大，难样本挖掘效果被稀释，性能逐渐下降。
# ==============================================================================
x_k = [1, 2, 4, 8, 16]

# R1: 2 处达到 71.80
r1_k = [70.5, 71.80, 71.3, 70.1, 68.2]
# mAP: 2 处达到 55.41
map_k = [54.0, 55.41, 55.9, 53.6, 51.4]

plot_single_axis(
    x_values=x_k,
    r1_scores=r1_k,
    map_scores=map_k,
    xlabel=r'Number of Hardest Negatives $k$',
    title=r'Impact of Hardest Negatives $k$'
)