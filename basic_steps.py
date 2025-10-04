import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
import json,time,os
import numpy as np

from tools import get_weight,get_bias,SELECT_data
from my_models import build_flexible_FNN

data_path = ' '
seed = 1
epochs = 200


#1.定义设备
def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device:{device}')
    return device
    

#2.定义数据预处理
def preprocess():
    transform = transforms.Compose([
    transforms.ToTensor(),#将图像转化为张量
    transforms.Normalize((0.5,),(0.5,)) #标准化图像数据
    ])
    return transform

#3.准备minist数据集
def prepare_minist(transform):
    train_dataset = datasets.MNIST('MNIST_data',train=True,download=True,transform=transform)
    test_dataset = datasets.MNIST('MNIST_data',train=False,download=True,transform=transform)
    #按需选取子集,需更改时替换新的选取函数
    train_dataset = SELECT_data(train_dataset)
    test_dataset = SELECT_data(test_dataset)
    return train_dataset,test_dataset

#4.根据batch size创建数据集加载器
def build_loader_by_batchsize(train_dataset,test_dataset,batchsize=40):
    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True)#打包训练集，每批64，随机抽取
    test_loader = DataLoader(test_dataset,batch_size=batchsize,shuffle=False)#打包测试集，每批64，不随机
    return train_loader,test_loader

#打包预处理，选取和创建加载器(上面3个功能)，按需选打包或分开
def preprocess_loader(batchsize):
    train_dataset,test_dataset = prepare_minist(preprocess())
    return build_loader_by_batchsize(train_dataset,test_dataset,batchsize)
#5.创建模型
#6.定义损失函数
#7.选择优化器

#全包干(上面7步)，懒人必备。
#输入:宽度(每层神经元数)，长度(层数)，batchsize(批大小),损失函数，优化器；
#输出:设备，模型实例，优化器，损失函数，训练/测试集加载器
def full_prepare(nl=10,num_hidden_layers=2,batchsize=40,criterion=nn.MultiMarginLoss(),Optimizer=optim.SGD):
    device = get_device()
    train_loader,test_loader = preprocess_loader(batchsize)
    model = build_flexible_FNN(nl,num_hidden_layers,device)
    optimizer = Optimizer(model.parameters(),lr=0.01)
    return device,model,optimizer,criterion,train_loader,test_loader

''' 一个epoch的训练。'''
'''依次传入: 设备，模型，优化器，损失函数，训练集；'''
'''传出数据记录:本次loss,实时总训练数,实时正确训练数'''
def train_one_epoch(device,model,optimizer,criterion,train_loader):
    running_loss = 0.0 #实时loss
    total_train = 0 #实时总训练数
    correct_train = 0  #实时正确训练数
    model.train()
    for inputs,labels in train_loader:#历遍训练集
        inputs,labels = inputs.to(device),labels.to(device)#获取图，标签，传入GPU
        optimizer.zero_grad()#清空历史梯度
        outputs,layer_output = model(inputs)#前向传播
        loss =criterion(outputs,labels)#计算损失函数值
        loss.backward()#计算梯度
        optimizer.step()#反向传播
        #记录数据
        running_loss += loss.item()
        _,predicted = torch.max(outputs,1)
        total_train += labels.size(0)
        correct_train += (predicted==labels).sum().item()
    return running_loss,total_train,correct_train,layer_output

'''一个epoch的测试。'''
'''依次传入: 设备，模型，训练集；''' 
'''传出数据记录：本次测试准确率'''      
def test_one_epoch(device,model,test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs,labels in test_loader:
        #步骤
            inputs,labels = inputs.to(device),labels.to(device)
            outputs,_ = model(inputs)
        #记录
            _,predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
            test_accuracy = correct/total
        return test_accuracy       

'''打包一个epoch的训练和测试,按需选打包或分开。'''
'''打包的附加功能：储存最优模型(权重/偏置)'''
'''依次传入: 设备，模型，优化器，损失函数，训练集,测试集；''' 
'''传出数据记录：'''        
def run_one_epoch(device,model,optimizer,criterion,train_loader,test_loader):
    t1 = time.time()
    best_accuracy = 0.0 #记录最佳验证集准确率
    #训练
    running_loss,total_train,correct_train,layer_output = train_one_epoch(device,model,optimizer,criterion,train_loader)   
    #计算训练时每轮准确率，损失值
    train_accuracy = correct_train/total_train
    train_loss = running_loss/len(train_loader)
    
    #测试
    test_accuracy = test_one_epoch(device,model,test_loader)
    #实时储存最佳权重
    if test_accuracy >best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(),os.path.join(data_path,'best_models',f'best_model(seed{seed}).pth'))
    #打印进度信息
    t2 = time.time()   
    return train_loss,train_accuracy,test_accuracy,layer_output,t2-t1

'''打包一次完整模拟，按需打包或分开。'''
'''依次传入：设备，模型，优化器，损失函数，训练集,测试集,epochs,seed''' 
'''功能1:储存epo概括信息,储存所有epoch权重,储存所有epoch偏置,储存所有epoch中间输出'''  
'''功能2:返回 最后准确率/最高准确率'''
def one_simulation(device,model,optimizer,criterion,train_loader,test_loader,epochs,seed):
    t1 = time.time()
    summaries = {'train_losses':[],'train_accuracies':[],'test_accuracies':[]}#所有epo概括信息
    weights = [get_weight(model),]#储存所有epoch权重
    biases = [get_bias(model),]#储存所有epoch偏置
    layer_outputs = []#储存所有epoch中间输出    
    torch.manual_seed(seed)
    for epoch in range(epochs):
        train_loss,train_accuracy,test_accuracy,layer_output,dt = run_one_epoch(device,model,optimizer,criterion,train_loader,test_loader)
        print(f'epoch{epoch+1/epochs}: test_accuracy:{test_accuracy} 用时{dt:.2f}秒')
        #添加新epoch的概括信息
        summaries['train_losses'].append(train_loss)
        summaries['train_accuracies'].append(train_accuracy)
        summaries['test_accuracies'].append(test_accuracy)
        #添加新epoch的每层输出
        layer_outputs.append(layer_output)
        #添加新epoch的权重/偏置矩阵
        weight,bias = get_weight(model),get_bias(model)
        weights.append(weight),biases.append(bias)
    
    #保存所有epoch概括信息   
    np.savez(os.path.join(data_path,'loss_accuracies_during_epoch',f'loss_accuracies(seed{seed}).npz'),**summaries)
    
    #保存所有epoch权重--形状{fc：(epo,后层cell数，前层cell数)}
    formated_wei = {}
    for fc in weight.keys():#交换内外层
        formated_wei[fc] = [zd[fc] for zd in weights]
    for fc,w_ls in formated_wei.items():#字典每个值逐一转化为数组
        formated_wei[fc] = np.stack([w.cpu().numpy() for w in w_ls])
    np.savez(os.path.join(data_path,'dinamic_weights',f'dyn_weights_array(seed{seed}).npz'),**formated_wei)
    
    #储存每轮偏置--形状{fc：(epo,后层cell数，前层cell数)}
    formated_bi = {}
    for k in bias.keys():
        formated_bi[k] = [zd[k] for zd in bias]
    for k,b_ls in formated_bi.items():
        formated_bi[k] = np.stack([b.cpu().numpy() for b in b_ls])
    np.savez(os.path.join(data_path,'dinamic_bias',f'dyn_bias_array(seed{seed}).npz'),**formated_bi)
    
    #储存每轮中间输出--形状(epo，fc,batch_size,cell数)
    layers_epo_array = np.stack(layer_outputs)
    np.save(os.path.join(data_path,'dinamic_layers_out',f'layers_out(seed{seed})_shape{layers_epo_array.shape}.npy'),layers_epo_array)
    
    t2 = time.time()
    return t2-t1
        

#构建结果储存目录
# best_models:储存最优，而非最后一个epoch,
# dinamic_weights：每个epo的权重矩阵
# dinamic_bias,每个epo的偏置矩阵
# dinamic_layers_out,每个epo的每层输出矩阵
# loss_accuracies_during_epoch，每个epo的loss，训练/测试的accuracies
def build_result_tree(data_path):
    data_list = ['best_models','dinamic_weights','dinamic_bias','dinamic_layers_out','loss_accuracies_during_epoch']
    for name in data_list:
        check_path = os.path.join(data_path,name)
        if not os.path.exists(check_path) and os.path.isdir(check_path):
            try:
                os.mkdir(check_path)
                print(f'建立结果目录：{check_path}')
            except:
                print(f'Failed building {check_path} due to Unknowing error.')
        else:
            print(f'结果目录存在：{check_path}')