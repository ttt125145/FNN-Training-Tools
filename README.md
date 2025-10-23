# FNN-Training-Tools

本分支为全连接神经网络（FNN）在 MNIST 数据集上的批量训练与分析工具，支持多复本、多参数组合的自动化实验，便于神经网络泛化能力、参数敏感性等研究。

## 目录结构

- `packages/`：核心功能模块（数据处理、模型构建、训练流程等）
- `Runs/`：批量实验脚本模板
- `copy0/` 等：每个复本的实验结果目录（代码生成）
- `MNIST_data/`：MNIST 原始数据
- `data_seed.npz`：保存的样本子集索引（代码生成）
- `heatmap.py`：结果可视化脚本
- `check.py`：数据子集选取与验证脚本

## 快速开始

1. **环境准备**
   - Python 3.9
   - 推荐使用 GPU
   - 依赖库：`torch`, `torchvision`, `numpy`, `matplotlib`

2. **生成数据子集索引**
   ```bash
   python check.py
   ```
   生成 `data_seed.npz`，保证多复本实验数据一致。

3. **运行批量训练**
   - 修改 `Runs/run0.py` 或 `templates_of_batch_copies.py`，设置参数后运行：
   ```bash
   python Runs/run0.py
   ```

4. **结果分析与可视化**
   - 使用 `heatmap.py` 绘制准确率热力图等。

## 主要模块说明

- `packages/basic_steps.py`：训练、测试、数据加载、结果保存等核心流程
- `packages/my_models.py`：灵活的全连接神经网络模型构建
- `packages/tools.py`：数据子集选取、参数组合生成、权重/偏置提取等工具函数

## 结果目录说明

每个复本（如 `copy0/`）下包含：
- `best_models/`：保存每个模型的最佳权重
- `dinamic_weights/`：每 epoch 的权重
- `dinamic_bias/`：每 epoch 的偏置
- `dinamic_layers_out/`：每 epoch 的中间层输出
- `loss_accuracies_during_epoch/`：每 epoch 的 loss/accuracy 记录

