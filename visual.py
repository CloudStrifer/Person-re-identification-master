import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

# 1. 准备数据
# 我照着你的截图输入了同样的 4x4 矩阵数据
data = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 0, 0]
])

# 2. 定制颜色
# 原图不是标准的纯红，而是偏“肉粉色”到“橘红色”的渐变。
# 为了最还原，我们定义两种具体的颜色：
# light_color: 对应 0 的颜色 (浅肉粉色)
# dark_color:  对应 1 的颜色 (橘红色)
colors = ['#FADbd8', '#E74C3C']  # 你可以修改这两个十六进制码来微调
my_cmap = ListedColormap(colors)

# 如果你想要连续的渐变效果（比如数据里有0.5），可以用下面这行代替上面两行：
# my_cmap = "Reds"

# 3. 绘制图形
plt.figure(figsize=(4, 4)) # 设置画布大小，正方形

ax = sns.heatmap(data,
            cmap=my_cmap,        # 使用我们定义的颜色
            annot=True,          # 关键参数：开启数值显示
            fmt='d',             # 数值格式：'d'代表整数 (如果是小数用 '.2f')
            cbar=False,          # 隐藏旁边的颜色条
            square=True,         # 强制每个格子是正方形
            xticklabels=False,   # 隐藏X轴刻度
            yticklabels=False,   # 隐藏Y轴刻度
            linewidths=0,        # 这种图通常没有明显的格线，设为0
            annot_kws={"size": 28, "color": "black"} # 设置数字的大小和颜色
            )

# 4. 添加边框 (可选)
# 原图外面有一圈黑框，heatmap默认边框不明显，我们可以手动加粗
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_edgecolor('black')

plt.tight_layout()
plt.show()