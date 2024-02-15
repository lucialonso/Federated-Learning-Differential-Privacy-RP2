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
    epsilon_array = [['0', '0'], ['500', '5'], ['1500', '13'], ['3000', '23'], ['5000', '34']]
    range_epsilon = ['0.1', '0.3', '0.5', '0.7', '1.0', '5.0', '10.0', '20.0', '30.0']
    plt.ylabel('Testing Accuracy %')
    plt.xlabel('Epsilon')
    idx = 0

    # y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_10.0.dat')
    # plt.plot(range(100), y, label='num_poison = 0')
    for epsilon, percentage in epsilon_array:
        y = openfile('./detail/accuracy_when_poison_{}.dat'.format(epsilon))
        plt.plot(range_epsilon, y, linestyle=list(linestyles.items())[idx][1], label='num_poison = {}, {}%'.format(epsilon, percentage))
        idx = idx + 1

    # plt.title('Mnist Gaussian Accuracy over Epsilon')
    plt.legend()
    plt.savefig('./detail/final_acc_eps-lines.png')

    plt.figure()
    epsilon_array = [['0', '0'], ['500', '5'], ['1500', '13'], ['3000', '23'], ['5000', '34']]
    range_epsilon = ['0.1', '0.3', '0.5', '0.7', '1.0', '5.0', '10.0', '20.0', '30.0']
    plt.ylabel('Testing Loss')
    plt.xlabel('Epsilon')
    idx = 0

    # y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_10.0.dat')
    # plt.plot(range(100), y, label='num_poison = 0')
    for epsilon, percentage in epsilon_array:
        y = openfile('./detail/loss_when_poison_{}.dat'.format(epsilon))
        plt.plot(range_epsilon, y, linestyle=list(linestyles.items())[idx][1], label='num_poison = {}, {}%'.format(epsilon, percentage))
        idx = idx + 1

    # plt.title('Mnist Gaussian Loss over Epsilon')
    plt.legend()
    plt.savefig('./detail/final_loss_eps-lines.png')

    plt.figure()
    epsilon_array = [['0', '0'], ['500', '5'], ['1500', '13'], ['3000', '23'], ['5000', '34']]
    range_epsilon = ['5.0', '10.0', '20.0', '30.0']
    plt.ylabel('Testing Loss')
    plt.xlabel('Epsilon')
    idx = 0
    # y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_10.0.dat')
    # plt.plot(range(100), y, label='num_poison = 0')
    for epsilon, percentage in epsilon_array:
        y = openfile('./detail/clear_loss_when_poison_{}.dat'.format(epsilon))
        plt.plot(range_epsilon, y, linestyle=list(linestyles.items())[idx][1], label='num_poison = {}, {}%'.format(epsilon, percentage))
        idx = idx + 1

    # plt.title('Mnist Gaussian Loss over Epsilon Close Up')
    plt.legend()
    plt.savefig('./detail/clear_final_loss_eps-lines.png')

    # plt.figure()
    # epsilon_array = [['0', '0'], ['500', '5'], ['1500', '13'], ['3000', '23'], ['5000', '34']]
    # range_epsilon = ['0.3', '0.5', '0.7', '1.0', '5.0', '10.0', '20.0', '30.0']
    # plt.ylabel('Testing Loss')
    # plt.xlabel('Epsilon')
    #
    # # y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_10.0.dat')
    # # plt.plot(range(100), y, label='num_poison = 0')
    # for epsilon, percentage in epsilon_array:
    #     y = openfile('./detail/cut_loss_when_poison_{}.dat'.format(epsilon))
    #     plt.plot(range_epsilon, y, label='num_poison = {}, {}%'.format(epsilon, percentage))
    #     idx = idx + 1
    #
    # plt.title('Mnist Gaussian Loss over Epsilon Close Up')
    # plt.legend()
    # plt.savefig('./detail/clear_cut_final_loss_eps.png')

    # plt.figure()
    # epsilon_array = [['0', '0'], ['500', '5'], ['1500', '13'], ['3000', '23'], ['5000', '34']]
    # range_epsilon = ['0.1', '0.3']
    # plt.ylabel('Testing Accuracy')
    # plt.xlabel('Epsilon')
    #
    # # y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_10.0.dat')
    # # plt.plot(range(100), y, label='num_poison = 0')
    # for epsilon, percentage in epsilon_array:
    #     y = openfile('./detail/cut_acc_when_poison_{}.dat'.format(epsilon))
    #     plt.plot(range_epsilon, y, label='num_poison = {}, {}%'.format(epsilon, percentage))
    #     idx = idx + 1
    #
    # plt.title('Mnist Gaussian Accuracy over Epsilon Close Up')
    # plt.legend()
    # plt.savefig('./detail/clear_cut_final_acc_eps.png')

    plt.figure()
    epsilon_array = [['0', '0'], ['500', '5'], ['1500', '13'], ['3000', '23'], ['5000', '34']]
    range_epsilon = ['0.1', '0.3', '0.5', '0.7', '1.0', '5.0']
    plt.ylabel('Testing Accuracy %')
    plt.xlabel('Epsilon')
    idx = 0

    # y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_10.0.dat')
    # plt.plot(range(100), y, label='num_poison = 0')
    for epsilon, percentage in epsilon_array:
        y = openfile('./detail/clear_acc_when_poison_{}.dat'.format(epsilon))
        plt.plot(range_epsilon, y, linestyle=list(linestyles.items())[idx][1], label='num_poison = {}, {}%'.format(epsilon, percentage))
        idx = idx + 1

    # plt.title('Mnist Gaussian Accuracy over Epsilon Close Up')
    plt.legend()
    plt.savefig('./detail/clear_final_acc_eps-lines.png')