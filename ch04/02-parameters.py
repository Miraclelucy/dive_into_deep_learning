import os

import torch
from torch import nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))

print('1.访问第二个全连接层的参数')
print(net[2].state_dict())  # 当通过Sequential类定义模型时，我们可以通过索引来访问模型的任意层
print(net[2].bias)  # 第二个神经网络层提取偏置
print(net[2].bias.data)  # 第二个神经网络层提取偏置的实际值
print(net[2].weight.grad is None)  # 由于我们还没有调用这个网络的反向传播，所以参数的梯度处于初始状态。

print('2.一次性访问所有参数')
print(*[(name, param.shape) for name, param in net[0].named_parameters()])  # 输入层的参数
print(*[(name, param.shape) for name, param in net.named_parameters()])


# 3.嵌套块的参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net


print('3.嵌套块的参数')
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
print(rgnet)
print(rgnet(X))
print(rgnet[0][1][0].bias.data)  # 访问第一个主要的块，其中第二个子块的第一层的偏置项


# 4.1内置的初始化器
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


print('4.1内置的初始化器')
net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])


# 4.2所有参数初始化为给定的常数
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


print('4.2所有参数初始化为给定的常数')
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])


# 4.3使用Xavier初始化方法初始化第一层，然后第二层初始化为常量值42
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


print('4.3使用Xavier初始化方法初始化第一层，然后第二层初始化为常量值42')
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight)
print(net[2].weight.data)


# 5.参数自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


print('5.参数自定义初始化')
net.apply(my_init)
print(net[0].weight[:2])

# 6.多个层间共享参数
# 我们需要给共享层一个名称，以便可以引用它的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                    nn.ReLU(), nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print('6.多个层间共享参数')
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 我们需要给共享层一个名称，以便可以引用它的参数。
print(net[2].weight.data[0] == net[4].weight.data[0])
