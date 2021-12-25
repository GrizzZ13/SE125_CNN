import string

import matplotlib.pyplot as plt


def load_from_acc():
    epoch_list = []
    acc_list = []
    with open("./acc.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            epoch_list.append(int(line[6:9]))
            acc_list.append(float(line[20:26]))

    return epoch_list, acc_list


def load_from_log():
    acc_list = []
    loss_list = []
    with open("./log.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            acc_list.append(float(line[32:38]))
            loss_list.append(float(line[19:24]))

    return acc_list, loss_list


if __name__ == '__main__':
    epoch_list, acc_list_1 = load_from_acc()
    acc_list_2, loss_list = load_from_log()

    plt.title("loss.png of epoch")
    plt.plot(epoch_list, loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    plt.title("accuracy.png of epoch")
    plt.plot(epoch_list, acc_list_1, color="green", label="testing accuracy")
    plt.plot(epoch_list, acc_list_2, color="red", label="training accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
