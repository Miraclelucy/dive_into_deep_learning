import os

import torch
from torch import nn
from torch.nn import functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载和保存张量
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load("x-file")
print(x2)

y = torch.zeros(4)
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
print(Y)

torch.save(net.state_dict(), 'mlp.params')

clone_net = MLP()
clone_net.load_state_dict(torch.load("mlp.params"))

Y_clone = clone_net(X)
print(Y_clone)
