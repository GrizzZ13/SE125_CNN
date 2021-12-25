import time

import torch
import torchvision
import torchvision.transforms as transforms
import ssl

# show some of the training images
import matplotlib.pyplot as plt
import numpy as np

# Convolutional Neural Network
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import matplotlib.pyplot as plt


def img_show(img):
    img = img / 2 + 0.5  # un-normalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# def zca_whitening(inputs):
#     inputs = inputs.cpu().numpy()
#     sigma = np.dot(inputs, inputs.T) / inputs.shape[1]  # inputs是经过归一化处理的，所以这边就相当于计算协方差矩阵
#     U, S, V = np.linalg.svd(sigma)  # 奇异分解
#     epsilon = 0.1  # 白化的时候，防止除数为0
#     ZCAMatrix = np.dot(np.dot(U, np.diag(1.0 / np.sqrt(np.diag(S) + epsilon))), U.T)  # 计算zca白化矩阵
#     return torch.from_numpy(np.dot(ZCAMatrix, inputs))  # 白化变换


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        # self.max_pool_1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(16, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool_2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu5 = nn.ReLU()

        self.max_pool_3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(9408, 1568)
        self.fc2 = nn.Linear(1568, 1568)
        self.fc3 = nn.Linear(1568, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        # x = self.max_pool_1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.max_pool_2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)

        x = self.max_pool_3(x)

        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)

    # load cifar-10 data set
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    batch_size = 500
    EPOCH = 100

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2, pin_memory=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    print(len(train_set))
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    # show images
    img_show(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

    net = Net()
    # use GPU
    net.to(device)

    # Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train
    time1 = time.time()
    running_loss_list = []
    epoch_list = []
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        print(epoch)
        running_loss = 0.0
        num = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # use GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # inputs = zca_whitening(inputs)

            # use CPU
            # inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            num += 1
        running_loss_list.append(running_loss / num)
        epoch_list.append(epoch)

    print('Finished Training')
    plt.plot(epoch_list, running_loss_list)
    plt.title("loss.png of epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss.png")
    plt.show()

    time2 = time.time()
    print(time2 - time1)

    # save the trained model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # test the model on test set
    data_iter = iter(test_loader)
    images, labels = data_iter.next()

    # print images
    img_show(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # load back the trained mdoel
    net = Net()
    net.load_state_dict(torch.load(PATH))
    # output
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
    # correctness
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
