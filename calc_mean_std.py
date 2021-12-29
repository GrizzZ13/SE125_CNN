import time

import torch
import torchvision
import torchvision.transforms as transforms
import ssl

import numpy as np

import torch.nn as nn

import torch.optim as optim

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # load cifar-10 data set
    ssl._create_default_https_context = ssl._create_unverified_context

    batch_size = 256

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2, pin_memory=True)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=0)

    print(len(train_set))

    # get some random training images
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    mm = torch.ones(256, 3, 32, 32)
    print(mm.shape)
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # use GPU
        inputs, labels = data[0], data[1]
        print("inputs")
        print(inputs.shape)
        mm = torch.cat([mm, inputs], 0)
        print(mm.shape)

    for i, data in enumerate(test_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        # use GPU
        inputs, labels = data[0], data[1]
        print("inputs")
        print(inputs.shape)
        mm = torch.cat([mm, inputs], 0)
        print(mm.shape)

    nb_samples = 0.
    # 创建3维的空列表
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    print(mm.shape)
    N, C, H, W = mm.shape[:4]
    mm = mm.view(N, C, -1)  # 将w,h维度的数据展平，为batch,channel,data,然后对三个维度上的数分别求和和标准差
    print(mm.shape)
    # 展平后，w,h属于第二维度，对他们求平均，sum(0)为将同一纬度的数据累加
    channel_mean += mm.mean(2).sum(0)
    # 展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
    channel_std += mm.std(2).sum(0)
    # 获取所有batch的数据
    nb_samples += N
    # 获取同一batch的均值和标准差
    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)
