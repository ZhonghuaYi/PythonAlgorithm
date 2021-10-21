import numpy as np

# 计算代价函数
# X包含有一列1,因为theta中有一个偏置项
def lost(X, y, theta):
    m = len(y)
    J = 0
    J = (np.transpose(X * theta - y) * (X * theta - y)) / (2 * m)
    return J
