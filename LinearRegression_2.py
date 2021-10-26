import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b ＋ 噪声"""
    X = torch.normal(-1 ,1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def write_csv(data, file_path, columns_index):
    import csv
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns_index)
        for d in data:
            writer.writerow(d)


def read_csv(file):
    import pandas as pd
    data_frame = pd.read_csv(file)
    data = torch.tensor(np.array(data_frame))
    return data.to(torch.float32)


# 不同于LinearRegression.py中的data_iter函数使用生成器来产生样本，这个函数直接使用DataSet与DataLoader产生数据
# DataLoader本身也是一个生成器
def data_load(data_arrays, batch_size, shuffle=True):
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # 读取数据
    data = read_csv('data/linear_regression.csv')
    features = data[:, 0:-1]
    labels = data[:, -1].unsqueeze(1)

    # 创建一个线性全连接层（全连接由Linear定义），（2，1）表示输入和输出特征形状
    net = nn.Sequential(nn.Linear(2, 1))
    # 将网络输入层的特征权重初始化，权重参数从（0， 0.01）的正态分布中随机采样
    net[0].weight.data.normal_(0, 0.01)
    # 将网络输入层的偏置初始化为0
    net[0].bias.data.fill_(0)

    # 定义损失函数为MSELoss，它计算网络的L2范数，默认情况下返回的是所有样本损失的平均值
    loss = nn.MSELoss()

    # 定义优化算法为SGD，其实例化时需要指定优化的参数（通过net.parameters()获得）以及超参数
    trainer = torch.optim.SGD(net.parameters(), lr=0.05)

    # 进行训练
    num_epochs = 3
    batch_size = 10
    data_iter = data_load((features, labels), batch_size)
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        
        l = loss(net(features), labels)
        print(f'epoch{epoch + 1}, loss: {l:f}')