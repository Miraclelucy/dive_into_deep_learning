import os

import torch
from torch import nn
from d2l import torch as d2l
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 多输入通道
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
print(corr2d_multi_in(X, K))

# 2. 多输出通道
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
K = torch.stack((K, K + 1, K + 2), 0)
print(K.shape)
print(corr2d_multi_in_out(X, K))

# 3. 1X1卷积层
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    print('c_i: ', c_i)  # 输入的通道数
    print('h: ', h)  # 输入的高
    print('w: ', w)  # 输入的宽
    c_o = K.shape[0]  # 卷积核的通道数
    X = X.view(c_i, h * w)  # 3 * 9
    K = K.view(c_o, c_i)  # 2 * 3
    Y = torch.mm(K, X)  # 全连接层的矩阵乘法
    return Y.view(c_o, h, w)

X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6

# (Y1 - Y2).norm().item() < 1e-6
