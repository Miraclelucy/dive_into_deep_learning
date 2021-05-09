import os
import sys

import torch
from torch import nn

sys.path.append('D:\\pythonspace\\d2l\\d2lutil')  # 加入路径，添加目录
import common

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3), torch.pow(features, 4)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.normal(0, 0.01, size=labels.size())
features[:2], poly_features[:2], labels[:2], features.size()  # 查看数据集的前两个样本
num_epochs = 100


def fit_and_plot(train_features, train_labels, test_features, test_labels):
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    net = nn.Linear(train_features.shape[-1], 1)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    train_ls, test_ls = [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.reshape(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(loss(net(train_features), train_labels.reshape(-1, 1)).item())
        test_ls.append(loss(net(test_features), test_labels.reshape(-1, 1)).item())
    common.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                    range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          'bias:', net.bias.data)


fit_and_plot(poly_features[:n_train, :3], labels[:n_train], poly_features[n_train:, :3], labels[n_train:])
fit_and_plot(poly_features[:n_train, :], labels[:n_train], poly_features[n_train:, :], labels[n_train:])
fit_and_plot(features[:n_train, :], labels[:n_train], features[n_train:, :], labels[n_train:])
