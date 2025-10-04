import basic_steps as bs

'''模板--复制粘贴使用'''
'''模拟单批次，使用不同参数(如网络宽度,高度,batch_size),训练n个神经网络'''


'''必需设置'''
data_path = ''#结果总路径
seed = 1        #起始种子
epochs = 200    #训练步数

'''按需预定义'''
# nl = 10,        #每层神经元
# num_hidden_layers=2, #隐藏层数
# batchsize = 40,     #输入数据分成小批次的大小
# criterion = nn.MultiMarginLoss() # 损失函数
# Optimizer = optim.SGD   #优化器

'''主流程'''
def __main__():
    bs.build_result_tree(data_path)
    for _ in range(500):#示例，根据需求修改循环
        device,model,optimizer,criterion,train_loader,test_loader = bs.full_prepare(   )#此括号传入循环的参数，以匹配项目要求
        bs.one_simulation(device,model,optimizer,criterion,train_loader,test_loader,epochs,seed)
        seed += 1
        
if __name__ == '__main__':
    __main__()
    