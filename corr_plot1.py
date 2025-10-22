import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

ns = [10, 20, 40, 80, 160]
bzs = [240, 480, 1000, 2000]

# 创建2行子图：第一行5个（固定n），第二行4个（固定batch_size）
fig, axes = plt.subplots(
    2, 5, figsize=(20, 8)
)  # 2行，每行5列（第二行最后一个位置空着）
fig.suptitle(
    "Weights Correlations (KDE) - Fixed n (Top) vs Fixed Batch Size (Bottom)",
    fontsize=16,
)

# 预加载所有数据到字典中，避免重复读取文件
data_dict = {}
for i, n in enumerate(ns):
    for j, bz in enumerate(bzs):
        f = f"cal/corr_nl{i}bz{j}.pt"
        if os.path.exists(f):
            data_dict[(i, j)] = torch.load(f).numpy()

# 第一行：固定n，变化batch_size（5个子图）
for i, n in enumerate(ns):
    ax = axes[0, i]  # 第一行，第i列

    # 绘制当前n值下所有batch_size的曲线
    for j, bz in enumerate(bzs):
        if (i, j) in data_dict:
            subplot_data = data_dict[(i, j)]
            sns.kdeplot(data=subplot_data, ax=ax, label=f"bz={bz}", alpha=0.7)

    ax.set_title(f"Fixed n={n} (All Batch Sizes)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

# 第二行：固定batch_size，变化n（4个子图）
for j, bz in enumerate(bzs):
    ax = axes[1, j]  # 第二行，第j列

    # 绘制当前batch_size下所有n值的曲线
    for i, n in enumerate(ns):
        if (i, j) in data_dict:
            subplot_data = data_dict[(i, j)]
            sns.kdeplot(data=subplot_data, ax=ax, label=f"n={n}", alpha=0.7)

    ax.set_title(f"Fixed batch_size={bz} (All n values)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

# 隐藏第二行最后一个空的子图（因为只有4个batch_size但布局是5列）
axes[1, 4].set_visible(False)

# 调整布局避免重叠
plt.tight_layout()
plt.show()
