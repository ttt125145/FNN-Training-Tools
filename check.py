import numpy as np 
from packages.tools import SELECT_data_seed,SELECT_indices
from torchvision import datasets,transforms

# transform = transforms.Compose([
# transforms.ToTensor(),#将图像转化为张量
# transforms.Normalize((0.5,),(0.5,)) #标准化图像数据
#     ])
# train_dataset = datasets.MNIST('MNIST_data',train=True,download=True,transform=transform)
# test_dataset = datasets.MNIST('MNIST_data',train=False,download=True,transform=transform)
# seed1 = SELECT_indices(train_dataset,200)
# seed2 = SELECT_indices(test_dataset,200)
# seed1 = np.array(seed1)
# seed2 = np.array(seed2)
# dict1 = {'train_seed':seed1,'test_seed':seed2}
# np.savez('data_seed.npz',**dict1)

SELECT_data_seed(200)

dicts = np.load('data_seed.npz')
print(dicts['train_seed'].shape)