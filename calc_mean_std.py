import time

import torch
import torchvision
import torchvision.transforms as transforms
import ssl

import numpy as np

import torch.nn as nn

import torch.optim as optim

if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # Assuming that we are on a CUDA machine, this should print a CUDA device:
    # print(device)
    #
    # # load cifar-10 data set
    # ssl._create_default_https_context = ssl._create_unverified_context
    #
    # batch_size = 256
    #
    # train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                          download=True, transform=transforms.ToTensor())
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
    #                                            shuffle=True, num_workers=2, pin_memory=True)
    #
    # print(len(train_set))
    #
    # # get some random training images
    # data_iter = iter(train_loader)
    # images, labels = data_iter.next()
    #
    # for i, data in enumerate(train_loader, 0):
    #     # get the inputs; data is a list of [inputs, labels]
    #     # use GPU
    #     inputs, labels = data[0].to(device), data[1].to(device)
    #     print("inputs")
    #     print(inputs)

    # 这里以上述创建的单数据为例子
    data = np.array([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        [[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]],
        [[5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5], [5, 5, 5]]
    ], dtype='uint8')

    # 将数据转为C,W,H，并归一化到[0，1]
    data = transforms.ToTensor()(data)
    # 需要对数据进行扩维，增加batch维度
    data = torch.unsqueeze(data, 0)

    nb_samples = 0.
    # 创建3维的空列表
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    print(data.shape)
    N, C, H, W = data.shape[:4]
    data = data.view(N, C, -1)  # 将w,h维度的数据展平，为batch，channel,data,然后对三个维度上的数分别求和和标准差
    print(data.shape)
    # 展平后，w,h属于第二维度，对他们求平均，sum(0)为将同一纬度的数据累加
    channel_mean += data.mean(2).sum(0)
    # 展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
    channel_std += data.std(2).sum(0)
    # 获取所有batch的数据，这里为1
    nb_samples += N
    # 获取同一batch的均值和标准差
    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)
