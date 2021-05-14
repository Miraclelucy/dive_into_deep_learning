import os

import torch
from torch import nn
from torch.nn import functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1.实例化nn.Sequential来构建我们的模型
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(),
                    nn.Linear(256, 10))
X = torch.rand(2, 20)
print('1.实例化nn.Sequential来构建我们的模型')
print(net(X))


# 2.自定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


net = MLP()
print('2.自定义模型')
print(net(X))


# 3.自定义顺序模型
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        print(args)  # 一个tuple, tuple和list非常类似，但是，tuple一旦创建完毕，就不能修改了。
        for block in args:
            # 这里，`block`是`Module`子类的一个实例。我们把它保存在'Module'类的成员变量
            # `_children` 中。`block`的类型是OrderedDict。
            self._modules[block] = block  # 每个Module都有一个_modules属性

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        print(self._modules.values())
        for block in self._modules.values():
            X = block(X)
        return X

print('3.自定义顺序模型')
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))


# 4.如何将任意代码集成到神经网络计算的流程中
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=True)
        self.liner = nn.Linear(20, 20)

    def forward(self, X):
        X = self.liner(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.liner(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


print('4.如何将任意代码集成到神经网络计算的流程中')
net = FixedHiddenMLP()
print(net(X))


# 5.组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

print('5.组合块')
chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
