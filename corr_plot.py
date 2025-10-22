import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
tensor = torch.load('cal/corr_nl0bz0.pt')
data = tensor.numpy().flatten()  # 确保是一维数据

plt.figure(figsize=(10, 6))

# 尝试不同的带宽参数
sns.kdeplot(data, fill=True, color='blue', alpha=0.6)  # 减小带宽放大细节
plt.title('Kernel Density Estimation (KDE) with Smaller Bandwidth')
plt.xlabel('Value')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()