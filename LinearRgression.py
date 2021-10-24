import random
import torch
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b ＋ 噪声"""
    X = torch.normal(0 ,1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(features[:, 0].detach().numpy(), features[:, 1].detach().numpy(), labels.detach().numpy())
    plt.show()




