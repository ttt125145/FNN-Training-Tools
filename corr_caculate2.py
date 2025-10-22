import torch
import numpy as np
import os
import gc
from itertools import combinations

def batch_pearson_correlation(x, y):
    """
    批量计算Pearson相关系数
    x, y: shape [batch_size, *feature_shape]
    返回: shape [batch_size]
    """
    assert x.shape == y.shape, "张量形状必须一致"
    
    batch_size = x.shape[0]
    
    # 重塑为 [batch_size, features]
    x_flat = x.reshape(batch_size, -1)
    y_flat = y.reshape(batch_size, -1)
    
    # 计算均值
    x_mean = torch.mean(x_flat, dim=1, keepdim=True)
    y_mean = torch.mean(y_flat, dim=1, keepdim=True)
    
    # 计算协方差和标准差
    cov = torch.mean((x_flat - x_mean) * (y_flat - y_mean), dim=1)
    std_x = torch.std(x_flat, dim=1, unbiased=False)
    std_y = torch.std(y_flat, dim=1, unbiased=False)
    
    # 避免除以零
    zero_std = (std_x == 0) | (std_y == 0)
    corr = cov / (std_x * std_y)
    corr[zero_std] = 0
    
    return corr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

data_path = 'E:/new_project/copy{c}/dinamic_weights/dyn_weights_array(seed{seed}).npz'
result_path = os.path.join(os.getcwd(), 'cal')
os.makedirs(result_path, exist_ok=True)

xiangdui_seeds = [
    [6, 12, 25, 50],
    [56, 62, 75, 100],
    [106, 112, 125, 150],
    [156, 162, 175, 200],
    [206, 212, 225, 250],
]

# 预计算所有组合
all_pairs = list(combinations(range(80), 2))
total_pairs = len(all_pairs)  # 3160
print(f"Total pairs to compute: {total_pairs}")

batch_size = 256  # 根据GPU内存调整批次大小

for x in range(3,5):  # 5个nl的循环
    nl = 10 * 2 ** x  # 修正：应该是2**x，不是2^x
    for y in range(4):  # 4个bz点的循环
        xseed = xiangdui_seeds[x][y]
        corr_tensor = torch.zeros(total_pairs, device='cpu')  # 结果存储在CPU上
        
        # 分批处理
        for batch_start in range(0, total_pairs, batch_size):
            batch_end = min(batch_start + batch_size, total_pairs)
            current_batch_size = batch_end - batch_start
            
            batch_w1 = []
            batch_w2 = []
            batch_indices = []
            
            # 准备批次数据
            for idx in range(batch_start, batch_end):
                i, j = all_pairs[idx]
                
                seed_i = 250 * i + xseed
                seed_j = 250 * j + xseed
                
                # 加载第一个权重文件
                f1_path = data_path.format(c=i, seed=seed_i)
                w1_data = np.load(f1_path)
                npw1 = np.stack(list(w1_data.values())[1][-1,:,:])
                batch_w1.append(npw1)
                
                # 加载第二个权重文件
                f2_path = data_path.format(c=j, seed=seed_j)
                w2_data = np.load(f2_path)
                npw2 = np.stack(list(w2_data.values())[1])
                batch_w2.append(npw2)
                
                batch_indices.append(idx)
                
                # 及时释放numpy数组
                del w1_data, w2_data
                gc.collect()
            
            # 批量转换为tensor并移到GPU
            batch_w1_np = np.stack(batch_w1)
            batch_w2_np = np.stack(batch_w2)
            
            tw1 = torch.from_numpy(batch_w1_np).to(device)
            tw2 = torch.from_numpy(batch_w2_np).to(device)
            
            # 批量计算相关性
            batch_corr = batch_pearson_correlation(tw1, tw2)
            
            # 将结果移回CPU并存储
            corr_tensor[batch_start:batch_end] = batch_corr.cpu()
            
            # 清理GPU内存
            del tw1, tw2, batch_corr
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # 清理CPU内存
            del batch_w1, batch_w2, batch_w1_np, batch_w2_np
            gc.collect()
            
            print(f'nl{x} bz{y}, batch {batch_start//batch_size + 1}/{(total_pairs-1)//batch_size + 1}, '
                  f'progress: {batch_end}/{total_pairs}')
        
        # 保存结果
        torch.save(corr_tensor, os.path.join(result_path, f'corr_nl{x}bz{y}.pt'))
        print(f'Saved results for nl{x} bz{y}')

print("All computations completed!")