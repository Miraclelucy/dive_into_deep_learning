import os
import sys

import torch
from d2l import torch as d2l
from torch import nn

sys.path.append('D:\\pythonspace\\d2l\\d2lutil')  # 加入路径，添加目录
import common

batch_size = 256
train_iter, test_iter = common.load_fashion_mnist(batch_size)

#初始化参数
num_inputs = 784
num_outputs = 10
num_hidden = 256

W1=torch.normal(0, 0.01, size=(num_inputs,num_hidden), requires_grad = True)
b1=torch.zeros(num_hidden, requires_grad = True)
W2=torch.normal(0, 0.01, size=(num_hidden,num_outputs), requires_grad = True)
b2=torch.zeros(num_outputs, requires_grad = True)

#激活函数采用ReLu

#定义模型
def net(X):
    X = X.reshape(-1,num_inputs)
    H = torch.relu(torch.matmul(X,W1) + b1)
    return torch.matmul(H,W2) + b2

#定义损失函数
loss = nn.CrossEntropyLoss()

#定义优化算法采用sgd

#训练模型
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()

#计算这个训练集的准确率
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


# 这里设置成100.0。(之所以这么大，应该是因为d2l里面的sgd函数在更新的时候除以了batch_size，
# 其实PyTorch在计算loss的时候已经除过一次了，sgd这里应该不用除了)
num_epochs, lr = 5, 100


# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            # 执行优化方法
            if optimizer is not None:
                optimizer.step()
            else:
                d2l.sgd(params, lr, batch_size)

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, [W1, b1, W2, b2], lr)
