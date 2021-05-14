import math
import os

import numpy as np
import torch
from d2l import torch as d2l

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n = 10000
a = torch.ones(n)
b = torch.ones(n)
c = torch.zeros(n)
timer = d2l.Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(c)
print("{0:.5f} sec".format(timer.stop()))

timer.start()
d = a + b
print(d)
print("{0:.5f} sec".format(timer.stop()))


def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp((- 0.5 / sigma ** 2) * (x - mu) ** 2)


## 可视化正态分布
x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x', ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
d2l.plt.show()
