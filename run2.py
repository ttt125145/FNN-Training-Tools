import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import packages.basic_steps as bs
from packages.tools import select_nlbz_pot

'''模板--复制粘贴使用'''
'''在单批次，使用不同参数(如网络宽度,高度,batch_size),训练n个神经网络的基础上,训练m个复本'''

'''必需设置'''
base_data_path = 'E:/new_project'#结果总路径
seed = 2001        #起始种子
epochs = 200    #训练步数
copy_num = 1    #复本数



'''按需预定义'''
# nl = 10,        #每层神经元
# num_hidden_layers=2, #隐藏层数
# batchsize = 40,     #输入数据分成小批次的大小
# criterion = nn.MultiMarginLoss() #损失函数
# Optimizer = optim.SGD   #优化器
#models_num =  #单批训练模型数

nl_list = [10,20,40,80,160]
bz_list = range(40,2001,40)
pot_arr = select_nlbz_pot(nl_list,bz_list)
'''主流程'''
def __main__():
    device= bs.get_device()
    for c in range(copy_num):
        #创建复本结果路径
        data_path = f'{base_data_path}/copy{c+2}'
        if not os.path.isdir(data_path):
            os.mkdir(data_path)
            print(f'建立复本结果目录{data_path}')
        else:
            print(f"复本结果路径{data_path}已存在")
        bs.build_result_tree(data_path)
        
        global seed
        for i,j in np.ndindex(pot_arr.shape[0],pot_arr.shape[1]):#示例，根据需求修改循环
            nl,bz = int(pot_arr[i,j,0]),int(pot_arr[i,j,1])#np.Int32转化为python内置int类型
            device,model,optimizer,criterion,train_loader,test_loader = bs.almost_prepare(#此括号传入循环的参数，以匹配项目要求
                device=device,
                nl=nl,
                num_hidden_layers=2,
                batchsize=bz,
                criterion=nn.MultiMarginLoss(),
                Optimizer=optim.SGD)   
            dt = bs.one_simulation(device,model,optimizer,criterion,train_loader,test_loader,epochs,seed,data_path)
            print(f'复本{c+2},model(nl{nl},bz{bz}),用时{dt/60:.2f}分')
            seed += 1

if __name__ == '__main__':
    __main__()    