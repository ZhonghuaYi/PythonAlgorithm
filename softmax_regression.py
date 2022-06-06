import torch
import torchvision
from torch.utils import data
from torchvision import transforms


# 读取FashionMNIST数据集，并返回数据集和验证集的迭代器
def load_fashion_mnist(batch_size, num_workers,resize=None):
    # trans为读取数据集后对图像进行的操作
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 读取FashionMNIST数据集，download表示是否需要下载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root='data', train=True, transform=trans, download=False
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root='data', train=False, transform=trans, download=False
    )
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=num_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=num_workers))

# 返回对应于一个数字或数字序列的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shit', 'trouser', 'pullover', 'dress', 'coat', 
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 对X每行特征分别取e指数，每行求和，再用X每行的原特征e指数除以原特征e指数的和
# 这样能够保证运算结果的特征均为正值，且特征值的和为1，符合概率特性
# 在计算过程中，可能会出现特征中有极大或者极小值，造成数值上溢或下溢
def softmax(X):
    X_exp = torch.exp(X)
    X_exp_sum = X_exp.sum(1, keepdim=True)
    return X_exp / X_exp_sum


# 定义模型
def net(X, w, b):
    return softmax(torch.matmul(X, w) + b)


# 定义损失函数（使用的是交叉熵损失函数）
def loss(y_hat, y):
    print(y)
    return - torch.log(y_hat[range(len(y_hat)), y])


# 定义优化方法
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 计算正确率
def accuracy(y_hat, y):
    # 如果y_hat是一个矩阵，那么每一行代表一个样本，每行中最大的那个值即对应的最大概率
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) # argmax返回的是最大值的索引
    cmp = y_hat.type(y.dtype) == y # 这里将y_hat的数据类型变为和y一样从而比较
    accurate_sum = float(cmp.type(y.dtype).sum()) # 正确的数目
    return accurate_sum / len(y)


if __name__ == '__main__':

    # 设定一些数据集参数
    batch_size = 20
    num_workers = 4 # dataloader读取数据使用的进程数

    # 设定模型参数
    num_inputs = 784 # fashionMNIST每张图片为28x28，展开为向量后是784
    num_outputs = 10 # fashionMNIST类别有十种，因此输出的向量为10（采用one hot编码）
    w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # 载入数据集
    train_iter, test_iter = load_fashion_mnist(batch_size, num_workers)

    # 定义训练参数并开始训练
    epochs = 3
    lr = 0.1
    for epoch in range(epochs):
        train_loss = 0.
        train_num = 0
        for X, y in train_iter:
            X.resize_(batch_size, int(X.numel() / batch_size))
            l = loss(net(X, w, b), y)
            break
            l.sum().backward()
            train_loss += l.sum()
            train_num += 1
            sgd([w, b], lr, batch_size)

        train_loss = train_loss / train_num
        with torch.no_grad():
            accurate_sum = 0.
            test_num = 0
            for X, y in test_iter:
                X.resize_(batch_size, int(X.numel() / batch_size))
                accurate_sum += accuracy(net(X, w, b), y)
                test_num += 1
            accurate_rate = accurate_sum / test_num
            print(f'epoch:{epoch+1}, accurate rate:{accurate_rate}, train loss:{train_loss}')
        break
            




