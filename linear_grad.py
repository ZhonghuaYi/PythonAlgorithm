"""
比较手动计算的导数（梯度）和pytorch的backward获得的导数（梯度）
"""
import torch

x = torch.normal(0, 1, (10, 1))
true_w = torch.tensor(2.)
true_b = torch.tensor(1.)
y_hat = x*true_w + true_b
y_hat += torch.normal(0, 0.01, y_hat.shape)

w = torch.tensor(0., requires_grad=True)
b = torch.tensor(0., requires_grad=True)
y = x*w + b
l = (y-y_hat)**2 / (2*len(y))

# 手动计算导数（梯度）
with torch.no_grad():
    w_grad = torch.sum(x*(y-y_hat)) / len(y)
    b_grad = torch.sum(y-y_hat) / len(y)
    print(f"w_grad:{w_grad}")
    print(f"b_grad:{b_grad}")

# 通过backward计算导数（梯度）
l.sum().backward()
print(f"w:{w.grad}")
print(f"b:{b.grad}")

