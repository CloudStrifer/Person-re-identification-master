import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格，使其看起来更像学术论文的图表
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['figure.dpi'] = 120 # 提高分辨率

# ==========================================
# Demo 1: 分析 A2DFP 中的平衡系数 λ (Lambda)
# ==========================================
# 模拟数据：假设 lambda 在 0.6 左右达到峰值
lambda_values = np.arange(0.0, 1.1, 0.1)
# 模拟 Rank-1 数据，先升后降
rank1_scores_lambda = [55.2, 57.5, 59.1, 60.8, 62.3, 63.1, 63.5, 62.8, 61.5, 59.0, 56.5]
# 模拟 mAP 数据，趋势相似但数值较低
map_scores_lambda = [48.1, 49.5, 51.2, 52.8, 54.1, 55.0, 55.3, 54.7, 53.5, 51.8, 49.2]

fig, ax1 = plt.subplots(figsize=(6, 4.5))

# 绘制 Rank-1 曲线 (左侧 Y 轴)
color_r1 = '#d62728' # 红色
line1 = ax1.plot(lambda_values, rank1_scores_lambda, marker='o', color=color_r1, 
                 linewidth=2, markersize=8, label='Rank-1')
ax1.set_xlabel(r'Balancing Coefficient $\lambda$ (in A2DFP)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Rank-1 Accuracy (%)', color=color_r1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_r1)
ax1.set_xticks(np.arange(0.0, 1.1, 0.2)) # 设置X轴刻度
ax1.grid(True, linestyle='--', alpha=0.6)

# 绘制 mAP 曲线 (右侧 Y 轴，双Y轴图在论文中很常见)
ax2 = ax1.twinx() 
color_map = '#1f77b4' # 蓝色
line2 = ax2.plot(lambda_values, map_scores_lambda, marker='s', color=color_map, 
                 linewidth=2, markersize=8, label='mAP')
ax2.set_ylabel('mAP (%)', color=color_map, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_map)

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best', fontsize=10)

plt.title(r'Sensitivity Analysis of $\lambda$', fontsize=14)
plt.tight_layout()
# plt.savefig('lambda_sensitivity.pdf') # 实际使用时保存为 PDF 矢量图
plt.show()


# ==========================================
# Demo 2: 分析 CDFP Loss 中的 Margin (m)
# ==========================================
# 模拟数据：文中设定为 0.1，我们假设在 0.1 处效果最好
margin_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
# 模拟 Rank-1 数据，在 0.1 处达到峰值
rank1_scores_margin = [60.5, 62.8, 64.2, 63.5, 61.0, 55.5]
# 模拟 mAP 数据
map_scores_margin = [53.2, 55.1, 56.8, 56.0, 53.5, 48.0]

fig, ax = plt.subplots(figsize=(6, 4.5))

# 这次用同一个Y轴绘制两条线
ax.plot(margin_values, rank1_scores_margin, marker='^', color='#2ca02c', 
        linewidth=2, markersize=8, label='Rank-1')
ax.plot(margin_values, map_scores_margin, marker='D', color='#ff7f0e', 
        linewidth=2, markersize=8, label='mAP')

ax.set_xlabel(r'Margin $m$ (in CDFP Loss)', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
ax.set_xticks(margin_values) # 确保 X 轴只显示我们测试的点
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='best', fontsize=10)

# 标注出最佳点 (论文中常用来强调默认设置)
best_m_idx = 2 # 对应 m=0.1
ax.annotate(f'Default $m=0.1$', 
            xy=(margin_values[best_m_idx], rank1_scores_margin[best_m_idx]),
            xytext=(margin_values[best_m_idx]+0.05, rank1_scores_margin[best_m_idx]+2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=10, ha='center')

plt.title(r'Impact of Margin $m$ on Performance', fontsize=14)
plt.tight_layout()
plt.show()


# ==========================================
# Demo 3: 分析 Loss 权重 (例如 λ2 for CDFP)
# ==========================================
# 模拟数据：通常在对数尺度上变化权重
loss_weights = [0.1, 0.5, 1.0, 2.0, 5.0]
# 模拟数据，假设 1.0 是最佳的
rank1_scores_w = [58.0, 61.5, 63.5, 62.0, 57.5]
map_scores_w = [50.0, 54.2, 56.0, 54.8, 50.5]

fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(loss_weights, rank1_scores_w, marker='*', color='#9467bd', 
        linewidth=2, markersize=10, label='Rank-1')
ax.plot(loss_weights, map_scores_w, marker='X', color='#8c564b', 
        linewidth=2, markersize=9, label='mAP')

# 关键点：设置 X 轴为对数尺度，因为权重的变化跨度较大
ax.set_xscale('log') 
ax.set_xlabel(r'Loss Weight $\lambda_2$ (CDFP)', fontsize=12, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')

# 自定义 X 轴的刻度标签，使其更易读
ax.set_xticks(loss_weights)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

ax.grid(True, linestyle='--', alpha=0.6, which='both') # both 包含次级网格线
ax.legend(loc='best', fontsize=10)

plt.title(r'Sensitivity to CDFP Loss Weight $\lambda_2$', fontsize=14)
plt.tight_layout()
plt.show()