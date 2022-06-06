import imp
import torch
from torch.utils import data
import torchvision
from torchvision import transforms


def load_mnist(batch_size, num_workers, resize=None):
    """读取MNIST数据集"""
    # 设置trans，即读取前对数据集的预操作
    trans = [transforms.ToTensor()]
    if resize:
        trans.append(transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # 读取数据集
    mnist_trans = torchvision.datasets.MNIST(
        root='data', train=True, transform=trans, download=False
    )
    mnist_test = torchvision.datasets.MNIST(
        root='data', train=False, transform=trans, download=False
    )
    # 生成数据集的loader，训练集打乱，测试集不需要打乱
    return (data.DataLoader(mnist_trans, batch_size, shuffle=True, 
                            num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=num_workers))
    
def softmax(X):
    X_exp = torch.exp(X)
    X_exp_sum = X_exp.sum(1, keepdim=True)
    return X_exp / X_exp_sum

def sigmiod(X):
    return 1 / (1 + torch.exp(-X))

def linear(X, w, b):
    return softmax(torch.matmul(X, w) + b)

def loss(Y_hat, Y):
    pass
    

if __name__ == "__main__":
    pass
