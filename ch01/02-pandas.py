import os

import numpy as np
import pandas as pd
import torch
from numpy import nan as NaN

os.makedirs(os.path.join('..', 'data'), exist_ok=True)  # 在上级目录创建data文件夹
datafile = os.path.join('..', 'data', 'house_tiny.csv')  # 创建文件
with open(datafile, 'w') as f:  # 往文件中写数据
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 第1行的值
    f.write('2,NA,106000\n')  # 第2行的值
    f.write('4,NA,178100\n')  # 第3行的值
    f.write('NA,NA,140000\n')  # 第4行的值

data = pd.read_csv(datafile)  # 可以看到原始表格中的空值NA被识别成了NaN
print('1.原始数据:\n', data)

inputs, outputs = data.iloc[:, 0: 2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())  # 用均值填充NaN
print(inputs)
print(outputs)
# 利用pandas中的get_dummies函数来处理离散值或者类别值。
# [对于 inputs 中的类别值或离散值，我们将 “NaN” 视为一个类别。] 由于 “Alley”列只接受两种类型的类别值 “Pave” 和 “NaN”
inputs = pd.get_dummies(inputs, dummy_na=True)
print('2.利用pandas中的get_dummies函数处理:\n', inputs)

x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print('3.转换为张量：')
print(x)
print(y)

# 扩展填充函数fillna的用法
df1 = pd.DataFrame([[1, 2, 3], [NaN, NaN, 2], [NaN, NaN, NaN], [8, 8, NaN]])  # 创建初始数据
print('4.函数fillna的用法：')
print(df1)
print(df1.fillna(100))  # 用常数填充 ，默认不会修改原对象
print(df1.fillna({0: 10, 1: 20, 2: 30}))  # 通过字典填充不同的常数，默认不会修改原对象
print(df1.fillna(method='ffill'))  # 用前面的值来填充
# print(df1.fillna(0, inplace=True))  # inplace= True直接修改原对象

df2 = pd.DataFrame(np.random.randint(0, 10, (5, 5)))  # 随机创建一个5*5
df2.iloc[1:4, 3] = NaN
df2.iloc[2:4, 4] = NaN  # 指定的索引处插入值
print(df2)
print(df2.fillna(method='bfill', limit=2))  # 限制填充个数
print(df2.fillna(method="ffill", limit=1, axis=1))  #
