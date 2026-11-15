import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 准备数据
# ==========================================

# 假设 alpha 从 0.0 到 1.0，步长为 0.1
# 你可以根据实际实验密度调整，例如 np.arange(0, 1.05, 0.05)
alphas = np.arange(0.0, 1.1, 0.1)

# beta 自动计算
betas = 1 - alphas

# 【在此处填入你的真实实验数据】
# 这里模拟了一组数据，假设在 alpha=0.3, beta=0.7 时效果最好
f1_scores = [0.8082, 0.9255, 0.9302, 0.9401, 0.9451, 0.9415, 0.9033, 0.8954, 0.8811, 0.8205, 0.8032]

# 确保数据长度一致
if len(alphas) != len(f1_scores):
    print(f"错误：alpha的数量 ({len(alphas)}) 与 F1分数数量 ({len(f1_scores)}) 不匹配")
    exit()

# ==========================================
# 2. 设置绘图风格 (学术论文风格)
# ==========================================

# 设置字体，优先使用 Times New Roman (如果系统有)，否则使用 DejaVu Serif
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 12

# 创建画布
plt.figure(figsize=(8, 6), dpi=120)

# ==========================================
# 3. 绘制折线图
# ==========================================

# 绘制线条和数据点
plt.plot(alphas, f1_scores,
         marker='o',        # 数据点形状：圆形
         linestyle='-',     # 线条风格：实线
         color='#1f77b4',   # 线条颜色：标准蓝
         linewidth=2,       # 线宽
         markersize=8,      # 数据点大小
         label='F1 Score')  # 图例标签

# ==========================================
# 4. 寻找并标注最优值 (Max F1)
# ==========================================

# 找到最大值的索引
max_index = np.argmax(f1_scores)
max_alpha = alphas[max_index]
max_f1 = f1_scores[max_index]

# 在图上用红点标出最大值
plt.plot(max_alpha, max_f1, marker='*', color='red', markersize=12, zorder=10)

# # 添加注释文本
# plt.annotate(f'Optimal\nα={max_alpha:.1f}, F1={max_f1:.3f}',
#              xy=(max_alpha, max_f1),
#              xytext=(max_alpha + 0.1, max_f1 + 0.02),
#              arrowprops=dict(facecolor='black', arrowstyle='->'),
#              fontsize=10)

# ==========================================
# 5. 设置坐标轴和标签
# ==========================================

plt.title('impact of parameters $\\alpha$ and $\\beta$ on the F1 performance of rosa roxburghii on the ViT-L32', fontsize=14, pad=15)
plt.ylabel('F1 Score', fontsize=12)

# X轴标签：特别设计，展示 alpha 和 beta 的关系
#plt.xlabel(r'Parameter $\alpha$ (with $\beta = 1 - \alpha$)', fontsize=12)

# 自定义 X 轴刻度标签，格式为 "alpha\n(beta)" 或者 "alpha (beta)"
# 这样读者可以同时看到两个参数的值
tick_labels = [f"{a:.1f}\n({1-a:.1f})" for a in alphas]
plt.xticks(alphas, tick_labels)

# 设置网格
plt.grid(True, linestyle='--', alpha=0.6)

# 设置Y轴范围，稍微留点余地让图好看一点
plt.ylim(min(f1_scores) - 0.05, max(f1_scores) + 0.1)

# ==========================================
# 6. 保存或显示
# ==========================================

plt.tight_layout()

# 保存为高清图片，通常论文需要 .pdf 或 .png (300 dpi以上)
# plt.savefig('parameter_sensitivity.pdf', format='pdf', bbox_inches='tight')
# plt.savefig('parameter_sensitivity.png', format='png', dpi=300, bbox_inches='tight')

print("绘图完成。显示图表...")
plt.show()