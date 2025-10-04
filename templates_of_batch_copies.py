import packages.basic_steps as bs
import torch.nn as nn
import torch.optim as optim
import os

'''模板--复制粘贴使用'''
'''在单批次，使用不同参数(如网络宽度,高度,batch_size),训练n个神经网络的基础上,训练m个复本'''

'''必需设置'''
base_data_path = ''#结果总路径
seed = 1        #起始种子
epochs = 200    #训练步数
copy_num = 3    #复本数



'''按需预定义'''
# nl = 10,        #每层神经元
# num_hidden_layers=2, #隐藏层数
# batchsize = 40,     #输入数据分成小批次的大小
# criterion = nn.MultiMarginLoss() #损失函数
# Optimizer = optim.SGD   #优化器
models_num = 500 #单批训练模型数


'''主流程'''
def __main__():
    for i in range(copy_num):
        #创建结果路径
        data_path = os.path.join(base_data_path,f'copy{i}')
        if not os.path.exists(data_path) and os.path.isdir(data_path):
            os.mkdir(data_path)
        else:
            print(f"复本结果路径{data_path}已存在")
        bs.build_result_tree(data_path)
        
        for j in range(500):#示例，根据需求修改循环
            device,model,optimizer,criterion,train_loader,test_loader = bs.full_prepare(#此括号传入循环的参数，以匹配项目要求
                nl=10,num_hidden_layers=2,
                batchsize=40,
                criterion=nn.MultiMarginLoss(),
                Optimizer=optim.SGD)   
            dt = bs.one_simulation(device,model,optimizer,criterion,train_loader,test_loader,epochs,seed)
            print(f'复本{i},model{j},用时{dt/60:.2f}分')
            seed += 1

if __name__ == '__main__':
    __main__()