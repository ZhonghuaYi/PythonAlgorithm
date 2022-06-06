from encodings import utf_8
import torch
import random
import numpy as np
from matplotlib import pyplot as plt


def data_generate(half_num, dim):
    """
    产生具有两个类别的数据

    Args:
        half_num (int): 一个类别的样本数
        dim (int): _一个类别的特征数

    Returns:
        (torch.tensor, torch.tensor): (2*half_num, dim)的X和(2*half_num, 1)的Y
    """
    num = int(half_num)
    X0 = torch.normal(-1, 0.5, (num, dim))
    Y0 = torch.zeros((num, 1))
    Z0 = torch.cat((X0, Y0), dim=1)
    X1 = torch.normal(1, 0.5, (num, dim))
    Y1 = torch.ones((num, 1))
    Z1 = torch.cat((X1, Y1), dim=1)
    Z = torch.cat((Z0, Z1), dim=0)
    return Z[:, :-1], Z[:, -1]


def data_iter(X, Y, batch_size):
    """
    产生batch_size大小的训练数据

    Args:
        X (torch.tensor): 特征
        Y (torch.tensor): 标签
        batch_size (int): 批量大小

    Yields:
        (torch.tensor, torch.tensor): X和Y的生成器
    """
    num = len(Y)
    ind = list(range(num))
    random.shuffle(ind)
    for i in range(0, num, batch_size):
        batch_index = ind[i:min(num, i + batch_size)]
        yield X[batch_index], Y[batch_index]


def data_draw(X, Y):
    X_0 = X[0:int(X.shape[0]/2), ...]
    X_1 = X[int(X.shape[0]/2):, ...]
    label = np.zeros(X_0.shape)
    plt.scatter(X_0.numpy(), label, c=plt.cm.Set1(0))
    plt.scatter(X_1.numpy(), label, c=plt.cm.Set1(1))
    plt.show()
    

def data_save(X, Y, file_name):
    data = torch.cat((X, Y.reshape(-1,1)), axis=1).numpy()
    np.savetxt(file_name, data, delimiter=",")
    

def data_load(file_name):
    data = np.loadtxt(file_name, delimiter=",")
    X = data[..., :-1]
    Y = data[..., -1]
    return torch.tensor(X).float(), torch.tensor(Y).float()


def sigmiod(X):
    """sigmiod函数"""
    return 1 / (1 + torch.exp(-X))


def linear(X, w, b):
    """线性模型"""
    return sigmiod(torch.matmul(X, w) + b)


def loss(Y_hat, Y):
    """交叉熵损失函数"""
    # return -Y * torch.log(Y_hat)
    return (Y - Y_hat)**2 / 100


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= param.grad * lr / batch_size
            param.grad.zero_()


def accurancy(Y_hat, Y):
    with torch.no_grad():
        predict = (Y_hat > 0.5).int().clone()
        predict = predict[:, 0]
        result = (predict == Y).int().clone()
        return result.sum() / len(Y)


if __name__ == "__main__":
    # X, Y = data_generate(500, 1)
    X0, Y0 = data_generate(50, 1)
    # data_save(X, Y, "data/classify.csv")
    w = torch.normal(1., 0.01, (1, 1), requires_grad=True)
    ws = [w.item()]
    b = torch.tensor(0., requires_grad=True)
    bs = [b.item()]
    
    print(w.item())
    print(b.item())
    
    X, Y = data_load("data/classify.csv")
    # data_draw(X, Y)

    lr = 0.01
    batch_size = 50
    epochs = 5
    
    plt.subplot(1, 2, 1)
    X_0 = X0[0:int(X0.shape[0]/2), ...]
    X_1 = X0[int(X0.shape[0]/2):, ...]
    label = np.zeros(X_0.shape)
    plt.scatter(X_0.numpy(), label, c=plt.cm.Set1(0))
    plt.scatter(X_1.numpy(), label, c=plt.cm.Set1(1))

    for epoch in range(epochs):
        for x, y in data_iter(X, Y, batch_size):
            y_hat = linear(x, w, b)
            l = loss(y_hat, y)
            l.sum().backward()
            sgd((w, b), lr, batch_size)
            ws.append(w.item())
            bs.append(b.item())
        with torch.no_grad():
            Y_hat = linear(X, w, b)
            l = loss(Y_hat, Y)
            a = accurancy(linear(X0, w, b), Y0)
            print(f"epoch{epoch}: \nloss:{l.mean()}\naccurancy:{a}\nw:{w}\n")

    with torch.no_grad():
        y0 = torch.matmul(X0, w) + b
        plt.plot(X0.numpy(), y0.numpy(), c=plt.cm.Set1(2))
        plt.plot(X0.numpy(), linear(X0, w, b).numpy(), c=plt.cm.Set1(3))
        plt.subplot(1,2,2)
        plt.scatter(np.array(range(len(ws))), np.array(ws), c=plt.cm.Set1(0))
        plt.scatter(np.array(range(len(bs))), np.array(bs), c=plt.cm.Set1(1))
        plt.show()
        print(f"w:{w}\nb:{b}")
        print(ws)