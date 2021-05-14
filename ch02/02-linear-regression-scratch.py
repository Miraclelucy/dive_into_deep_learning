import random

import torch

## with torch.no_grad() 则主要是用于停止autograd模块的工作，
## 以起到加速和节省显存的作用，具体行为就是停止gradient计算，从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。

## mm只能进行矩阵乘法,也就是输入的两个tensor维度只能是( n × m ) (n\times m)(n×m)和( m × p ) (m\times p)(m×p)
## bmm是两个三维张量相乘, 两个输入tensor维度是( b × n × m )和( b × m × p ), 第一维b代表batch size，输出为( b × n × p )
## matmul可以进行张量乘法, 输入可以是高维.

## python知识补充：
## Python3 range() 函数返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表。
## Python3 list() 函数是对象迭代器，可以把range()返回的可迭代对象转为一个列表，返回的变量类型为列表。
## Python3 range(start, stop[, step])
## Python3 shuffle() 方法将序列的所有元素随机排序。shuffle()是不能直接访问的，需要导入 random 模块。举例：random.shuffle (list)
## Python3 yield是python中的生成器


## 人造数据集
def create_data(w, b, nums_example):
    X = torch.normal(0, 1, (nums_example, len(w)))
    y = torch.matmul(X, w) + b
    print("y_shape:", y.shape)
    y += torch.normal(0, 0.01, y.shape)  # 加入噪声
    return X, y.reshape(-1, 1)  # y从行向量转为列向量


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = create_data(true_w, true_b, 1000)


## 读数据集
def read_data(batch_size, features, lables):
    nums_example = len(features)
    indices = list(range(nums_example))  # 生成0-999的元组，然后将range()返回的可迭代对象转为一个列表
    random.shuffle(indices)  # 将序列的所有元素随机排序。
    for i in range(0, nums_example, batch_size):  # range(start, stop, step)
        index_tensor = torch.tensor(indices[i: min(i + batch_size, nums_example)])
        yield features[index_tensor], lables[index_tensor]  # 通过索引访问向量


batch_size = 10
for X, y in read_data(batch_size, features, labels):
    print("X:", X, "\ny", y)
    break;

##初始化参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 定义模型
def net(X, w, b):
    return torch.matmul(X, w) + b


# 定义损失函数
def loss(y_hat, y):
    # print("y_hat_shape:",y_hat.shape,"\ny_shape:",y.shape)
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2  # 这里为什么要加 y_hat_shape: torch.Size([10, 1])  y_shape: torch.Size([10])


# 定义优化算法
def sgd(params, batch_size, lr):
    with torch.no_grad():  # with torch.no_grad() 则主要是用于停止autograd模块的工作，
        for param in params:
            param -= lr * param.grad / batch_size  ##  这里用param = param - lr * param.grad / batch_size会导致导数丢失， zero_()函数报错
            param.grad.zero_()  ## 导数如果丢失了，会报错‘NoneType’ object has no attribute ‘zero_’


# 训练模型
lr = 0.03
num_epochs = 3

for epoch in range(0, num_epochs):
    for X, y in read_data(batch_size, features, labels):
        f = loss(net(X, w, b), y)
        # 因为`f`形状是(`batch_size`, 1)，而不是一个标量。`f`中的所有元素被加到一起，
        # 并以此计算关于[`w`, `b`]的梯度
        f.sum().backward()
        sgd([w, b], batch_size, lr)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print("w {0} \nb {1} \nloss {2:f}".format(w, b, float(train_l.mean())))

print("w误差 ", true_w - w, "\nb误差 ", true_b - b)
