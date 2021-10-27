import torch
import torchvision
from torch.utils import data
from torchvision import transforms


# 返回对应于一个数字或数字序列的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shit', 'trouser', 'pullover', 'dress', 'coat', 
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


if __name__ == '__main__':

    trans = transforms.ToTensor()

    # 读取FashionMNIST数据集，download表示是否需要下载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root='data', train=True, transform=trans, download=False
    )
    minst_test = torchvision.datasets.FashionMNIST(
        root='data', train=False, transform=trans, download=False
    )

