import torch
import torchvision
import torchvision.transforms as transforms
import ssl
from cnn import Net
from resnet import ResNet18

if __name__ == "__main__":
    tester = input("input a model you wanna test, 'elexnet' or 'resnet18'?\nInput q to quit.\n")
    while True:
        if tester == "elexnet":
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

            batch_size = 256

            test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                      shuffle=False, num_workers=2)
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            PATH = "./trained_model/model_elexnet.pth"

            # test the model on test set
            data_iter = iter(test_loader)
            images, labels = data_iter.next()

            # load back the trained model
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
        elif tester == "resnet18":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Assuming that we are on a CUDA machine, this should print a CUDA device:
            print(device)

            # load cifar-10 data set
            ssl._create_default_https_context = ssl._create_unverified_context
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            batch_size = 256

            test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                    download=True, transform=transform)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                      shuffle=False, num_workers=2)
            classes = ('plane', 'car', 'bird', 'cat',
                       'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

            PATH = "./trained_model/model_resnet.pth"

            # test the model on test set
            data_iter = iter(test_loader)
            images, labels = data_iter.next()

            # load back the trained model
            net = ResNet18()
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
        elif tester == "q":
            break

        tester = input("input 'elexnet', 'resnet18' or 'q'\n")
