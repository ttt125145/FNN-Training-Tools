import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import json, time, os
import numpy as np


# 选取每类num_per_class的子集。传入数据集，每类数。返回子集，选取索引列表。
def SELECT_data(dataset, num_per_class=200):
    """
    从数据集中为每个类别选取指定数量的样本，并返回一个子集

    参数:
    dataset: torch.utils.data.Dataset - 原始数据集
    num_per_class: int - 每个类别选取的样本数量

    返回:
    torch.utils.data.Subset - 选取后的子集
    """

    class_indices = {
        i: [] for i in range(10)
    }  # 创建一个字典，用于存储每个类别的样本索引

    # 遍历数据集，将每个样本的索引按类别存储
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected_indices = []  # 存储最终选取的样本索引
    # 对每个类别，随机选取指定数量的样本索引
    for class_id, indices in class_indices.items():
        selected_indices.extend(np.random.choice(indices, num_per_class, replace=False))

    return Subset(dataset, selected_indices)  # 返回一个Subset对象，包含选取的样本


# 索引版本
def SELECT_indices(dataset, num_per_class):
    """
    从数据集中为每个类别选取指定数量的样本索引，并返回这些索引

    参数:
    dataset: torch.utils.data.Dataset - 原始数据集
    num_per_class: int - 每个类别选取的样本数量

    返回:
    list - 选取的样本索引列表
    """

    class_indices = {
        i: [] for i in range(10)
    }  # 创建一个字典，用于存储每个类别的样本索引

    # 遍历数据集，将每个样本的索引按类别存储
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    selected_indices = []  # 存储最终选取的样本索引
    # 对每个类别，随机选取指定数量的样本索引
    for class_id, indices in class_indices.items():
        selected_indices.extend(np.random.choice(indices, num_per_class, replace=False))

    return selected_indices  # 返回选取的样本索引列表


# 同上，但是保存选取的索引列表为npz。形状：{'train_seed':arr1,'test_seed':arr2}。用于多个复本索引相同数据。
def SELECT_data_seed(num_per_class):
    """
    为训练集和测试集的每个类别选取指定数量的样本索引，并将这些索引保存为npz文件

    参数:
    num_per_class: int - 每个类别选取的样本数量

    返回:
    None

    该函数会在当前目录下生成一个名为"data_seed.npz"的文件，包含训练集和测试集的选取索引
    结构为: {'train_seed': arr1, 'test_seed': arr2}
    """

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # 将图像转化为张量
            transforms.Normalize((0.5,), (0.5,)),  # 标准化图像数据
        ]
    )

    # 加载 MNIST 训练集和测试集
    train_dataset = datasets.MNIST(
        "MNIST_data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "MNIST_data", train=False, download=True, transform=transform
    )

    # 为训练集和测试集选取样本索引
    seed1 = SELECT_indices(train_dataset, num_per_class)
    seed2 = SELECT_indices(test_dataset, num_per_class)

    # 将索引转换为 numpy 数组
    seed1 = np.array(seed1)
    seed2 = np.array(seed2)

    # 将索引保存为 npz 文件，方便后续加载和复用
    np.savez("data_seed.npz", **{"train_seed": seed1, "test_seed": seed2})


# 加载数据集
def load_seed_data(transform):
    """
    从 'data_seed.npz' 文件中加载预先保存的样本索引，并据此创建训练集和测试集子集

    参数:
    transform: torchvision.transforms.Compose - 用于数据预处理的转换操作

    返回:
    train_dataset: torch.utils.data.Subset - 训练集子集
    """

    dict = np.load("data_seed.npz")  # 加载保存的索引文件

    # 加载原始 MNIST 训练集和测试集
    train_dataset = datasets.MNIST(
        "MNIST_data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        "MNIST_data", train=False, download=True, transform=transform
    )

    # 使用加载的索引创建训练集和测试集子集
    train_dataset = Subset(train_dataset, dict["train_seed"].tolist())
    test_dataset = Subset(test_dataset, dict["test_seed"].tolist())

    return train_dataset, test_dataset


# 选取(神经元个数n，batchsize)参数组合的点。按range()语法传入二者选点元组，返回numpy数组(n点数，batchsize点数)
def select_pot(n_range=(10, 101, 10), batchsize_range=(40, 2001, 40)):
    """
    选取神经元数量和批次大小的参数组合点

    参数:
    n_range: tuple - 神经元数量的范围，格式为 (start, end, step)
    batchsize_range: tuple - 批次大小的范围，格式为 (start, end, step)

    返回:
    numpy.ndarray - 包含所有参数组合点的数组，形状为 (num_n_points, num_batchsize_points, 2)

    其中最后一个维度包含 (神经元数量, 批次大小)
    """

    n_pot = range(
        n_range[0], n_range[1], n_range[2]
    )  # 根据 n_range 生成神经元个数的序列
    b_pot = range(
        batchsize_range[0], batchsize_range[1], batchsize_range[2]
    )  # 根据 batchsize_range 生成批次大小的序列
    # 使用 meshgrid 生成所有 n 和 batchsize 的组合
    X, Y = np.meshgrid(n_pot, b_pot, indexing="ij")
    # 将 X 和 Y 堆叠起来，形成 [n, batchsize] 的组合数组
    return np.stack([X, Y], axis=-1)


# 功能同上,传入二者列表即可
def select_nlbz_pot(nl_list, bz_list):
    """
    选取神经元数量和批次大小的参数组合点

    参数:
    nl_list: list - 神经元数量的列表
    bz_list: list - 批次大小的列表

    返回:
    numpy.ndarray - 包含所有参数组合点的数组，形状为 (num_nl_points, num_bz_points, 2)
    """

    # 使用 meshgrid 生成所有 nl 和 bz 的组合
    X, Y = np.meshgrid(nl_list, bz_list, indexing="ij")

    # 将 X 和 Y 堆叠起来，形成 [nl, bz] 的组合数组
    return np.stack([X, Y], axis=-1)


# 获取权重/偏置矩阵
def get_weight(model):
    """
    获取模型的权重矩阵

    参数:
    model: nn.Module - 需要提取权重的神经网络模型

    返回:
    dict - 包含权重参数的字典，键为参数名称，值为对应的权重张量
    """

    w_dict = model.state_dict()  # 获取模型的 state_dict，包含所有参数
    weight_dict = {
        k: v for k, v in w_dict.items() if "bias" not in k
    }  # 过滤掉偏置项，只保留权重

    return weight_dict


def get_bias(model):
    """
    获取模型的偏置矩阵

    参数:
    model: nn.Module - 需要提取偏置的神经网络模型

    返回:
    dict - 包含偏置参数的字典，键为参数名称，值为对应的偏置张量
    """

    w_dict = model.state_dict()  # 获取模型的 state_dict，包含所有参数
    bias_dict = {
        k: v for k, v in w_dict.items() if "weight" not in k
    }  # 过滤掉权重项，只保留偏置

    return bias_dict
