from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib as mpl

def openfile(filepath):
    file = open(filepath)
    y = []
    while 1:
        line = file.readline()
        if line.rstrip('\n') == '':
            break
        y.append(float(line.rstrip('\n')))
        if not line:
            break
        pass
    file.close()
    return y

# ('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_Gaussian_epsilon_{}.dat'.format(epsilon))


linestyles = OrderedDict(
    [('solid',               (0, ())),
     ('densely dotted',      (0, (1, 1))),

     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('densely dashdotted',  (0, (3, 1, 1, 1)))])

# linestyle=list(linestyles.items())[idx][1]

if __name__ == '__main__':
    plt.figure()
    epsilon_array = ['0.1', '0.3', '0.5', '0.7', '1.0', '5.0', '10.0', '20.0', '30.0']
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Global Round')
    idx = 0
    # y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_10.0.dat')
    # plt.plot(range(100), y, label='num_poison = 0')
    for epsilon in epsilon_array:
        y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_{}.dat'.format(epsilon))
        plt.plot(range(100), y, label=r'$\epsilon={}$'.format(epsilon))
        idx = idx + 1

    plt.title('Mnist Gaussian')
    plt.legend()
    plt.savefig('./acc/mnist_gaussian_acc_eps.png')

    plt.figure()
    epsilon_array = ['0.1', '0.3', '0.5', '0.7', '1.0', '5.0', '10.0', '20.0', '30.0']
    plt.ylabel('Testing Loss')
    plt.xlabel('Global Round')
    idx = 0

    for epsilon in epsilon_array:
        y = openfile('./loss/lossfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_{}.dat'.format(epsilon))
        plt.plot(range(100), y, linestyle=list(linestyles.items())[idx][1], label=r'$\epsilon={}$'.format(epsilon))
        idx = idx + 1
    # y = openfile('./loss/lossfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_10.0.dat')
    # plt.plot(range(100), y, label='num_poison = 0')
    plt.title('Mnist Gaussian')
    plt.legend()
    plt.savefig('./loss/mnist_gaussian_loss_eps.png')