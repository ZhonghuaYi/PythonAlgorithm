import random
import torch
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b ＋ 噪声"""
    X = torch.normal(-1 ,1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 1, y.shape)
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
    return data


if __name__ == '__main__':
    # 由实际的w与b产生训练数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # # 将数据写入csv文件
    # data = torch.cat((features, labels), axis=1).numpy()
    # os.makedirs(os.path.join('data'), exist_ok=True)
    # file_path = os.path.join('data', 'linear_regression.csv')
    # columns_index = ['X[0]', 'X[1]', 'y']
    # write_csv(data, file_path)

    # # 从csv文件中读取数据
    # data = read_csv('data/linear_regression.csv')
    # features = data[:, 0:-1]
    # labels = data[:, -1]

    # 由X的前两组特征与X对应的标签y画出三维图
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(features[:, 0].detach().numpy(), features[:, 1].detach().numpy(), labels.detach().numpy())
    plt.show()




