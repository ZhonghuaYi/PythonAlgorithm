import random
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b ＋ 噪声"""
    # X = torch.normal(4, 1.4, (num_examples, len(w)))
    X = torch.normal(-1, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y


def write_csv(data, file_path, columns_index):
    import csv
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns_index)
        for d in data:
            writer.writerow(d)


def read_csv(file):
    import pandas as pd
    data_frame = pd.read_csv(file)
    data = torch.tensor(np.array(data_frame))
    return data.to(torch.float32)


# 这是一个生成器
def data_iter(batch_size, features, labels):
    num_examples = len(labels)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i +
                                                   batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 线性回归模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 均方损失函数
def squard_loss(y_hat, y):
    return (y_hat - y)**2 / 2


# 定义优化算法为小批量随机梯度下降
def sgd(params, lr, batch_size):
    """params为需要拟合的参数，lr为学习率"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    # 由实际的w与b产生训练数据
    true_w = torch.tensor([3.3, 2.1])
    true_b = torch.tensor(1.4)
    features, labels = synthetic_data(true_w, true_b, 50)
    print(features.shape, labels.shape)

    # 将数据写入csv文件
    # data = torch.cat((features, labels.reshape(-1,1)), axis=1).numpy()
    # os.makedirs(os.path.join('data'), exist_ok=True)
    # file_path = os.path.join('data', 'linear_regression.csv')
    # columns_index = ['X[0]', 'X[1]', 'y']
    # write_csv(data, file_path, columns_index)

    # # 从csv文件中读取数据
    # data = read_csv('data/linear_regression.csv')
    # features = data[:, 0:-1]
    # labels = data[:, -1]

    # # 由X的前两组特征与X对应的标签y画出三维图
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(features[:, 0].detach().numpy(), features[:, 1].detach().numpy(), labels.detach().numpy())
    # plt.show()

    # 初始化模型参数
    # w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    w = torch.normal(0, 0.1, true_w.shape, requires_grad=True)
    b = torch.tensor(0., requires_grad=True)
    # w = torch.tensor([1., -1.], requires_grad=True)
    # b = torch.tensor([0.], requires_grad=True)

    lr = 0.1
    num_epochs = 5
    net = linreg
    loss = squard_loss
    batch_size = 50

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # X与y的小批量损失
            print(l)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')

    print(f'true_w:{true_w}, true_b:{true_b}')
    print(f'w:{w}, b:{b}')
    print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差：{true_b - b}')
