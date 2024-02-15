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
    epsilon_array = [['5000', '50'], ['3000', '30'], ['1500', '15'], ['500', '5']]

    epsilon_array_two = ['1.0', '5.0', '10.0', '20.0', '30.0']
    plt.ylabel('Testing Loss')
    plt.xlabel('Global Round')
    idx = 0

    for epsilon, numpois in epsilon_array:
        y = openfile('./report/loss_after_1.0_poison_{}.dat'.format(epsilon))
        plt.plot(epsilon_array_two, y, linestyle=list(linestyles.items())[idx][1], label='num_poison = {}, {}%'.format(epsilon, numpois))
        idx = idx + 1
    y = openfile('./report/loss_after_1.0_poison_0.dat')
    plt.plot(epsilon_array_two, y, label='num_poison = 0')
    plt.title('Mnist Gaussian')
    plt.legend()
    plt.savefig('./report/loss-from-1.0.png')