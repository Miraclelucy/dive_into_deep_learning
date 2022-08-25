import os
import torch
from torch import nn
from d2l import torch as d2l
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 二维互相关运算函数
def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape # 卷积核的大小
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)) # 输出矩阵大小
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum() # X是输入矩阵
    return Y

# 调用二维互相关运算函数
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
Z = corr2d(X, K)
print(Z)

# 2. 二维卷积层
# 卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size)) # 卷积核权重
        self.bias = nn.Parameter(torch.zeros(1)) # 标量偏置

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 3. 二维卷积层的简单应用
# 首先，构造一个6*8像素的黑白图像
X1 = torch.ones((6, 8))
X1[:, 2:6] = 0
print(X1)
# 再构造一个高度为1、宽度为2的卷积核
K1 = torch.tensor([[1.0, -1.0]])
print(K1)

# 输出Y1中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为0。
Y1 = corr2d(X1, K1)
print(Y1)

# 现在我们将输入的二维图像转置，再进行如上的互相关运算
# 之前检测到的垂直边缘消失了，这个卷积核K只可以检测垂直边缘，无法检测水平边缘
X2 = X1.t()
Y2 = corr2d(X2, K1)
print(Y2)

# 4. 学习卷积核
# 当有了更复杂数值的卷积核，或者连续的卷积层时，我们不可能手动设计滤波器。那么我们是否可以学习由X生成Y的卷积核
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X1.reshape((1, 1, 6, 8))
Y = Y1.reshape((1, 1, 6, 7))
lr = 3e-2  # 学习率

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    # 迭代卷积核
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')

# 我们所学的卷积核的权重张量
print(conv2d.weight.data.reshape((1, 2)))
