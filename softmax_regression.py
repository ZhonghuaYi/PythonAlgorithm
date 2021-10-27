import torch
import torchvision
from torch.utils import data
from torchvision import transforms

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root='data', train=True, transform=trans, download=False
)

minst_test = torchvision.datasets.FashionMNIST(
    root='data', train=False, transform=trans, download=False
)

print(len(mnist_train), len(minst_test))
