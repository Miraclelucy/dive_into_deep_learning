# 查看pytorch中的所有函数名或属性名
import torch

print(dir(torch.distributions))

print('1.张量的创建')
# ones 函数创建一个具有指定形状的新张量，并将所有元素值设置为 1
t = torch.ones(4)
print('t:', t)

x = torch.arange(12)
print('x:', x)
print('x shape:', x.shape)  # 访问向量的形状

y = x.reshape(3, 4)  # 改变一个张量的形状而不改变元素数量和元素值
print('y:', y)
print('y.numel():', y.numel())  # 返回张量中元素的总个数

z = torch.zeros(2, 3, 4)  # 创建一个张量，其中所有元素都设置为0
print('z:', z)

w = torch.randn(2, 3, 4)  # 每个元素都从均值为0、标准差为1的标准高斯（正态）分布中随机采样。
print('w:', w)

q = torch.tensor([[1, 2, 3], [4, 3, 2], [7, 4, 3]])  # 通过提供包含数值的 Python 列表（或嵌套列表）来为所需张量中的每个元素赋予确定值
print('q:', q)

print('2.张量的运算')
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)  # **运算符是求幂运算
print(torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print('cat操作 dim=0', torch.cat((X, Y), dim=0))
print('cat操作 dim=1', torch.cat((X, Y), dim=1))  # 连结（concatenate） ,将它们端到端堆叠以形成更大的张量。

print('X == Y', X == Y)  # 通过 逻辑运算符 构建二元张量
print('X < Y', X < Y)
print('张量所有元素的和:', X.sum())  # 张量所有元素的和

print('3.广播机制')
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print('a:', a)
print('b:', b)
print('a + b:', a + b)  # 神奇的广播运算

print('4.索引和切片')
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
print('X:', X)
print('X[-1]:', X[-1])  # 用 [-1] 选择最后一个元素
print('X[1:3]:', X[1:3])  # 用 [1:3] 选择第二个和第三个元素]

X[1, 2] = 9  # 写入元素。
print('X:', X)

X[0:2, :] = 12  # 写入元素。
print('X:', X)

print('5.节约内存')
before = id(Y)  # id()函数提供了内存中引用对象的确切地址
Y = Y + X
print(id(Y) == before)

before = id(X)
X += Y
print(id(X) == before)  # 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销。

before = id(X)
X[:] = X + Y
print(id(X) == before)  # 使用 X[:] = X + Y 或 X += Y 来减少操作的内存开销。

print('6.转换为其他 Python对象')
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
A = Y.numpy()
print(type(A))  # 打印A的类型
print(A)
B = torch.tensor(A)
print(type(B))  # 打印B的类型
print(B)

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))
