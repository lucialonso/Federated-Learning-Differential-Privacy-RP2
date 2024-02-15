import os

import matplotlib.pyplot as plt
import matplotlib as mpl

# ('./log/accfile_fed_mnist_cnn_100_iidFalse_dp_Gaussian_epsilon_{}.dat'.format(epsilon))

# accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_0.1.dat

if __name__ == '__main__':
    rootpathacc = './report'
    if not os.path.exists(rootpathacc):
        os.makedirs(rootpathacc)
    # epsilon_array = ['0.1', '0.3', '0.5', '0.7', '1.0', '5.0']
    # epsilon_array = ['0.1', '0.3', '0.5', '0.7', '1.0', '5.0', '10.0', '20.0', '30.0']
    # epsilon_array = ['0.3', '0.5', '0.7', '1.0', '5.0', '10.0', '20.0', '30.0']
    # epsilon_array = ['0.1', '0.3']
    epsilon_array = ['1.0', '5.0', '10.0', '20.0', '30.0']

    # epsilon_array = ['5.0', '10.0', '20.0', '30.0']
    final_file = open(rootpathacc + '/loss_after_1.0_poison_5000.dat', "w")
    for epsilon in epsilon_array:
        accfile = open('./loss/lossfile_fed_poison_5000_cnn_100_iidFalse_dp_Gaussian_epsilon_{}.dat'.
                       format(epsilon), "r")
        # accfile = open('./loss/lossfile_fed_poison_5000_cnn_100_iidFalse_dp_Gaussian_epsilon_{}.dat'.
        #                format(epsilon), "r")
        lines = accfile.readlines()
        # first = lines[0].split(',')[0]
        end = lines[-1].split(',')[0]
        final_file.write(end)
        # final_file.write('\n')
        accfile.close()

    final_file.close()
