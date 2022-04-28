from pickletools import optimize
import numpy as np
import torch
import random


def data_generate(true_w, true_b, num):
    X = torch.normal(4, 1.4, (num, len(true_w)))
    Y = torch.matmul(X, true_w) + true_b
    Y += torch.normal(0, 0.01, Y.shape)
    return X, Y


def data_iter(X, Y, batch_size):
    # 生成随机的索引序列
    nums = len(Y)
    index = list(range(nums))
    random.shuffle(index)
    # 根据索引序列和batch_size，生成小批量的训练数据
    for i in range(0, nums, batch_size):
        batch_index = index[i:min(i + batch_size, nums)]
        yield X[batch_index], Y[batch_index]


def linear(X, w, b):
    """通过线性模型生成拟合值。

    Args:
        X (tensor): 特征
        w (tensor): 线性模型的权重
        b (tensor): 线性模型的偏置

    Returns:
        tensor: 线性模型的拟合值
    """
    return torch.matmul(X, w) + b


def square_loss(y, y_hat):
    """计算模型的均方损失。

    Args:
        y (tensor): 模型的拟合值
        y_hat (tensor): 标签值

    Returns:
        tensor: 返回的是均方损失
    """
    return (y - y_hat)**2 / 2


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == "__main__":
    # 设置实际的权重和偏置
    true_w = torch.tensor([3.3, 2.1])
    true_b = torch.tensor(1.4)
    # 通过实际的权重和偏置生成具有高斯噪声的训练数据
    example_num = 50
    X, Y_hat = data_generate(true_w, true_b, example_num)
    print(X.shape, Y_hat.shape)
    # 设置初始的用于训练的权重和偏置，并设置requires_grad为True
    w = torch.normal(0, 0.1, true_w.shape, requires_grad=True)
    b = torch.tensor(0., requires_grad=True)
    # 设置模型训练的参数
    lr = 0.07
    epoch_num = 5
    batch_size = 50
    net = linear
    loss = square_loss
    # 进行训练
    for epoch in range(epoch_num):
        for x, y in data_iter(X, Y_hat, batch_size):
            l = loss(net(x, w, b), y)
            print(l)
            l.sum().backward()
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_l = loss(net(X, w, b), Y_hat).sum()
            print(f"epoch{epoch+1}:{train_l}")

    print(f"true_w:{true_w}, true_b:{true_b}")
    print(f"w:{w}, b:{b}")
    print(f"误差:w:{true_w-w}, b:{true_b-b}")
