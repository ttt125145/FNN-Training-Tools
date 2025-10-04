项目名称
 
[FNN Training Tools] - [灵活地用参数控制打包好的功能，训练需要的神经网络，目前只有FNN]
 
 
 
项目介绍
 
 直接在项目下创建训练模型的代码，复制粘贴两个模板，按需修改。
 
 
 
目录结构说明
 
[TRAIN_TOOLS]/
├── __init__.py
├── basic_steps.py      #一些不同程度打包好的基础步骤，从准备设备，数据集，到训练模型，按需导入需要的功能
├── my_models.py        #储存常用的神经网络模型或灵活定义神经网络的功能
├── templates_of_batch_copies.py    #多复本训练流程模板，复制粘贴使用
├── templates_of_one_batch_simulations.py   #单复本训练流程模板，复制粘贴使用
├── tools.py         # 自定义的方便函数
└── README.md     # 项目说明文档（当前文件）
 
 
 
 
前置准备
 
1. conda环境准备
 
- 安装 [anaconda]，打开[anaconda_prompt]
- 创建项目环境：
- conda create [环境名] python=3.9
- 进入：
- conda activate [环境名]
2. 安装依赖库：
-conda install torch torchview numpy matplotlib
 
 
3. 克隆项目到本地:
- mkdir && cd [项目名]
- git clone  https://github.com/ttt125145/FNN-Training-Tools.git
   