import os

import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


X = torch.rand(3, 3, 3)
K = torch.rand(2, 3, 1, 1)
print('X: ', X)
print('K: ', K)
Y1 = corr2d_multi_in_out_1x1(X, K)
print('Y1: ', Y1)
# Y2 = corr2d_multi_in_out(X, K)

# (Y1 - Y2).norm().item() < 1e-6
