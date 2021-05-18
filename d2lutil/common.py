import torch
import torchvision
import torchvision.transforms as transforms
from d2l import torch as d2l
from torch.utils import data
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.datasets.utils import extract_archive


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()


def hello():
    print("semilogy_HELLO")

# 李沐课件中采用的是远程获取的方式，因为公司网络的限制，远程获取会报错。
# 运行远程获取的方式，c:\users\lwx898760\miniconda3\envs\d2l\lib\site-packages\torchvision\datasets\mnist.py会报错
# 这里采用本地下载的方式先将数据集下载到本地，放在D://d2l-data//下面
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
def load_fashion_mnist(batch_size):
    extract_archive('D://d2l-data//t10k-images-idx3-ubyte.gz', 'D://d2l-data//FashionMNIST//raw', False)
    extract_archive('D://d2l-data//train-images-idx3-ubyte.gz', 'D://d2l-data//FashionMNIST//raw', False)
    extract_archive('D://d2l-data//t10k-labels-idx1-ubyte.gz', 'D://d2l-data//FashionMNIST//raw', False)
    extract_archive('D://d2l-data//train-labels-idx1-ubyte.gz', 'D://d2l-data//FashionMNIST//raw', False)

    training_set = (
        read_image_file('D://d2l-data//FashionMNIST//raw//train-images-idx3-ubyte'),
        read_label_file('D://d2l-data//FashionMNIST//raw//train-labels-idx1-ubyte')
    )
    test_set = (
        read_image_file('D://d2l-data//FashionMNIST//raw//t10k-images-idx3-ubyte'),
        read_label_file('D://d2l-data//FashionMNIST//raw//t10k-labels-idx1-ubyte')
    )
    with open('D://d2l-data//FashionMNIST//processed//training.pt', 'wb') as f:
        torch.save(training_set, f)
    with open('D://d2l-data//FashionMNIST//processed//test.pt', 'wb') as f:
        torch.save(test_set, f)
    print('Done!')

    #train_data, train_targets = torch.load('D://d2l-data//FashionMNIST//processed//training.pt')
    #test_data, test_targets = torch.load('D://d2l-data//FashionMNIST//processed//test.pt')

    mnist_train = torchvision.datasets.FashionMNIST(root="D:/d2l-data/", train=True, transform=transforms.ToTensor(),
                                                    download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="D:/d2l-data/", train=False, transform=transforms.ToTensor(),
                                                   download=False)

    # 这里有个坑 如果线程数num_workers设置大于0会报错  An attempt has been made to start a new process before the current process has finished its bootstrapping
    train_iter = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return (train_iter, test_iter)
