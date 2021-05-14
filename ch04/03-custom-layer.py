import os

import torch
from torch import nn
from torch.nn import functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


print('1.不带参数的层')
layer = CenteredLayer()
print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
print(net)
Y = net(torch.rand(4, 8))  # Y是4*128维的
print(Y.mean())


# 带参数的层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


print('2.带参数的层')
dense = MyLinear(5, 3)
print(dense.weight)

Y = dense(torch.rand(2, 5))
print(Y)

net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))
