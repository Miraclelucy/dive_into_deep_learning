import torch

print('1.标量与变量')
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y, x * y, x / y, x ** y)

x = torch.arange(4)
print('2.向量')
print('x:', x)
print('x[3]:', x[3])  # 通过张量的索引来访问任一元素
print('张量的形状:', x.shape)  # 张量的形状
print('张量的长度:', len(x))  # 张量的长度
z = torch.arange(24).reshape(2, 3, 4)
print('三维张量的长度:', len(z))

print('3.矩阵')
A = torch.arange(20).reshape(5, 4)
print('A:', A)
print('A.shape:', A.shape)
print('A.shape[-1]:', A.shape[-1])
print('A.T:', A.T)  # 矩阵的转置

print('4.矩阵的计算')
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
print('A:', A)
print('B:', B)
print('A + B:', A + B)  # 矩阵相加
print('A * B:', A * B)  # 矩阵相乘

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print('X:', X)
print('a + X:', a + X)  # 矩阵的值加上标量
print('a * X:', a * X)
print((a * X).shape)

print('5.矩阵的sum运算')
print('A:', A)
print('A.shape:', A.shape)
print('A.sum():', A.sum())
print('A.sum(axis=0):', A.sum(axis=0))  # 沿0轴汇总以生成输出向量
print('A.sum(axis=1):', A.sum(axis=1))  # 沿1轴汇总以生成输出向量
print('A.sum(axis=1, keepdims=True)', A.sum(axis=1, keepdims=True))  # 计算总和保持轴数不变
print('A.sum(axis=[0, 1]):', A.sum(axis=[0, 1]))  # Same as `A.sum()`
print('A.mean():', A.mean())
print('A.sum() / A.numel():', A.sum() / A.numel())

print('6.向量-向量相乘（点积）')
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print('x:', x)
print('y:', y)
print('向量-向量点积:', torch.dot(x, y))

print('7.矩阵-向量相乘(向量积)')
print('A:', A)  # 5*4维
print('x:', x)  # 4*1维
print('torch.mv(A, x):', torch.mv(A, x))

print('8.矩阵-矩阵相乘(向量积)')
print('A:', A)  # 5*4维
B = torch.ones(4, 3)  # 4*3维
print('B:', B)
print('torch.mm(A, B):', torch.mm(A, B))

print('9.范数')
u = torch.tensor([3.0, -4.0])
print('向量的𝐿2范数:', torch.norm(u))  # 向量的𝐿2范数
print('向量的𝐿1范数:', torch.abs(u).sum())  # 向量的𝐿1范数
v = torch.ones((4, 9))
print('v:', v)
print('矩阵的𝐿2范数:', torch.norm(v))  # 矩阵的𝐿2范数

print('10.根据索引访问矩阵')
y = torch.arange(10).reshape(5, 2)
print('y:', y)
index = torch.tensor([1, 4])
print('y[index]:', y[index])

print('11.理解pytorch中的gather()函数')
a = torch.arange(15).view(3, 5)
print('11.1二维矩阵上gather()函数')
print('a:', a)
b = torch.zeros_like(a)
b[1][2] = 1  ##给指定索引的元素赋值
b[0][0] = 1  ##给指定索引的元素赋值
print('b:', b)
c = a.gather(0, b)  # dim=0
d = a.gather(1, b)  # dim=1
print('d:', d)
print('11.2三维矩阵上gather()函数')
a = torch.randint(0, 30, (2, 3, 5))
print('a:', a)
index = torch.LongTensor([[[0, 1, 2, 0, 2],
                           [0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1]],
                          [[1, 2, 2, 2, 2],
                           [0, 0, 0, 0, 0],
                           [2, 2, 2, 2, 2]]])
print(a.size() == index.size())
b = torch.gather(a, 1, index)
print('b:', b)
c = torch.gather(a, 2, index)
print('c:', c)
index2 = torch.LongTensor([[[0, 1, 1, 0, 1],
                            [0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]],
                           [[1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0]]])
d = torch.gather(a, 0, index2)
print('d:', d)

print('12.理解pytorch中的max()和argmax()函数')
a = torch.tensor([[1, 2, 3], [3, 2, 1]])
b = a.argmax(1)
c = a.max(1)
d = a.max(1)[1]
print('a:', a)
print('a.argmax(1):', b)
print('a.max(1):', c)
print('a.max(1)[1]:', d)

print('13.item()函数')
a = torch.Tensor([1, 2, 3])
print('a[0]:', a[0])  # 直接取索引返回的是tensor数据
print('a[0].item():', a[0].item())  # 获取python number
