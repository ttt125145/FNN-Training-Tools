import torch
import numpy as np
import os,gc


def pearson_correlation(x, y):
    # 确保输入形状相同
    assert x.shape == y.shape, "张量形状必须一致"
    
    # 计算均值
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    
    # 计算协方差和标准差
    cov = torch.mean((x - x_mean) * (y - y_mean))
    std_x = torch.std(x)
    std_y = torch.std(y)
    
    # 避免除以零
    if std_x == 0 or std_y == 0:
        return torch.tensor(0.0)
    
    return cov / (std_x * std_y)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device:{device}')

data_path = 'E:/new_project/copy{c}/dinamic_weights/dyn_weights_array(seed{seed}).npz'
result_path = os.path.join(os.getcwd(),'cal')

xiangdui_seeds = [[6,12,25,50],
                 [56,62,75,100],
                 [106,112,125,150],
                 [156,162,175,200],
                 [206,212,225,250],
                 ]

for x in range(3,5):#5个nl的循环
    nl = 10 * 2^x
    for y in range(4):#4个bz点的循环
        xseed = xiangdui_seeds[x][y]
        corr_tensor = torch.empty(3160)
        corr_index = 0
        for i in range(80):
            for j in range(80):
                if j > i :
                    seed_i = 250*i + xseed
                    seed_j = 250*j + xseed
                    
                    f1_path = data_path.format(c=i,seed=seed_i)
                    f2_path = data_path.format(c=j,seed=seed_j)
                    
                    w1 = np.load(f1_path,mmap_mode='r')
                    w2 = np.load(f2_path,mmap_mode='r')
                    npw1 = np.stack(list(w1.values())[1])
                    npw2 = np.stack(list(w2.values())[1])
                    tw1 = torch.from_numpy(npw1).to(device)
                    tw2 = torch.from_numpy(npw2).to(device)
                    corr = pearson_correlation(tw1,tw2)
                    corr_tensor[corr_index] = corr
                    corr_index += 1
                    del w1,w2,npw1,npw2,tw1,tw2
                    gc.collect()
                    print(f'nl{x}bz{y},{corr_index}/3160,{corr}')
        torch.save(corr_tensor,os.path.join(result_path,f'corr_nl{x}bz{y}.pt'))

                    
                    