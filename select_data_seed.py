from packages.tools import SELECT_data_seed
from packages.basic_steps import preprocess
from torchvision import datasets,transforms

transform = preprocess()
train_dataset = datasets.MNIST('MNIST_data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST('MNIST_data',train=False,download=True,transform=transform)

seed = SELECT_data_seed()