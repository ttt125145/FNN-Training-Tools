import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Subset
import json,time,os
import numpy as np

#选取每类num_per_class的子集。传入数据集，每类数。返回子集，选取索引列表。
def SELECT_data(dataset,num_per_class=200):
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    selected_indices = []
    for class_id, indices in class_indices.items():
        selected_indices.extend(np.random.choice(indices, num_per_class, replace=False)) 
    return Subset(dataset, selected_indices)

#选取(神经元个数n，batchsize)参数组合的点。按range()语法传入二者选点元组，返回numpy数组 (n点数，batchsize点数)
def select_pot(n_range=(10,101,10),batchsize_range=(40,2001,40)):
    n_pot = range(n_range[0],n_range[1],n_range[2])
    b_pot = range(batchsize_range[0],batchsize_range[1],batchsize_range[2])
    X,Y = np.meshgrid(n_pot,b_pot,indexing='ij')
    return np.stack([X,Y],axis=-1)


#获取权重/偏置矩阵
def get_weight(model):
    w_dict = model.state_dict()
    weight_dict = {k:v for k,v in w_dict.items() if 'bias'not in k}
    return weight_dict
def get_bias(model):
    w_dict = model.state_dict()
    bias_dict = {k:v for k,v in w_dict.items() if 'weight'not in k}
    return bias_dict


