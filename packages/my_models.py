import torch
import torch.nn as nn

'''此模块储存常用的 神经网络模型 ，或 按需创建神经网络的功能 ，待日渐完善。 '''



def build_flexible_FNN(nl, num_hidden_layers=2, device='cpu'):
    """
    构建具有任意隐藏层的全连接神经网络
    
    参数:
    nl: int - 每个隐藏层的神经元数量
    num_hidden_layers: int - 隐藏层的数量 (默认为2)
    device: str - 模型运行的设备 ('cpu' 或 'cuda')
    
    返回:
    model: nn.Module - 构建的神经网络模型
    """
    class FlexibleNet(nn.Module):
        def __init__(self, input_size=28 * 28, output_size=10):
            super().__init__()
            self.input_size = input_size
            self.output_size = output_size            
            # 创建隐藏层列表
            self.hidden_layers = nn.ModuleList()            
            # 输入层到第一个隐藏层
            self.hidden_layers.append(nn.Linear(input_size, nl))            
            # 中间隐藏层
            for _ in range(1, num_hidden_layers):
                self.hidden_layers.append(nn.Linear(nl, nl))            
            # 输出层
            self.output_layer = nn.Linear(nl, output_size)            
            # 激活函数
            self.activation = nn.ReLU()
                    
        def forward(self, x):
            x = torch.flatten(x, start_dim=1)            
            # 存储所有隐藏层的输出
            hidden_outputs = []            
            # 通过所有隐藏层
            for i, layer in enumerate(self.hidden_layers):
                x = layer(x)
                if i < len(self.hidden_layers) - 1:  # 除了最后一层隐藏层
                    x = self.activation(x)
                hidden_outputs.append(x)    
            # 输出层（无激活函数）
            output = self.output_layer(x)            
            return output, torch.stack(hidden_outputs, dim=0)
    
    # 创建模型并移动到指定设备
    model = FlexibleNet().to(device)
    return model