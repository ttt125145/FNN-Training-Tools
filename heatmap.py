import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


base_data_path = 'E:/new_project'
# 加载数据
start_seed = 1
accuracies_arr = np.zeros((3,250))
for c in range(3):
    for seed in range(start_seed,start_seed+250):
        data_path = f'{base_data_path}/copy{c}'
        summaries = np.load(f'{base_data_path}/copy{c}/loss_accuracies_during_epoch/loss_accuracies(seed{seed}).npz')
        last_accuracy = summaries['test_accuracies'][-1]
        accuracies_arr[c,seed-start_seed] = last_accuracy
    start_seed += 1000
average_accu = np.mean(accuracies_arr,axis=0)
average_nlbz_arr = average_accu.reshape(5,50)
copy0_accu = accuracies_arr[0,:]
copy0_nlbz_arr = copy0_accu.reshape(5,50)


"""平均曲线"""
# plt.figure(figsize=(12,6))
# plt.style.use("seaborn-v0_8")
# plt.xlabel('batch_size')
# plt.ylabel('accuracies')
# plt.title('copy0 accuracies change among batch_size')
# y = np.array([10,20,40,80,160])
# x = np.array(range(40, 2001, 40))
# for i in range(5):
#     bzs = copy0_nlbz_arr[i,:]
#     plt.plot(x,bzs,label=f'n={y[i]}')
# plt.legend()
# plt.savefig('copy0_accu_plot_in_different_n.png')
# plt.close()

"""平均热力图"""
# 创建坐标点
y = np.array([10,20,40,80,160])
y = np.log2(y/10)
x = np.array(range(40, 2001, 40))

# 创建自定义颜色映射
colors = [(1, 1, 1), (0, 1, 0), (0, 0, 1), (0.5, 0, 0.5), (1, 0, 0), (0, 0, 0)]  # 白->绿->蓝->紫->红->黑
color_map = LinearSegmentedColormap.from_list(name='MM', colors=colors, N=256)

# 设置绘图风格
plt.style.use("seaborn-v0_8")
plt.figure(figsize=(12, 8))

# 使用 imshow 绘制热力图
extent = [x.min(), x.max(), y.min(), y.max()]
heatmap = plt.imshow(copy0_nlbz_arr,  
                     origin='lower',
                     aspect='auto',
                     extent=extent,
                     cmap=color_map)

# 添加颜色条
cbar = plt.colorbar(heatmap)
cbar.set_label('Accuracy')

# 设置坐标轴标签和标题
plt.xlabel('batch_size')
plt.ylabel('log2(n/10)')
plt.title('copy0 accuracies under different batch_size and n')

# 设置坐标轴刻度
plt.xticks(np.arange(40, 2001, 200))  # 每200个batch_size一个刻度
plt.yticks(y)
#plt.yscale('log',base=2)    

# 添加网格线
#plt.grid(True, linestyle='--', alpha=0.5)

# 保存图像
plt.savefig('copy0_nlbz_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()