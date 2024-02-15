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



if __name__ == '__main__':
    plt.figure()
    epsilon_array = [['500', '5'], ['1500', '13'], ['3000', '23'], ['5000', '34']]
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Global Round')
    y = openfile('./acc/accfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_1.0.dat')
    plt.plot(range(100), y, label='num_poison = 0')
    for epsilon, percentage in epsilon_array:
        y = openfile('./acc/accfile_fed_poison_{}_cnn_100_iidFalse_dp_Gaussian_epsilon_1.0.dat'.format(epsilon))
        plt.plot(range(100), y, label='num_poison = {}, {}%'.format(epsilon, percentage))

    plt.title('Mnist Gaussian')
    plt.legend()
    plt.savefig('./acc/mnist_gaussian_acc_eps_1.0.png')

    plt.figure()
    epsilon_array = [['5000', '50'], ['3000', '30'], ['1500', '15'], ['500', '5']]
    plt.ylabel('Testing Loss')
    plt.xlabel('Global Round')

    for epsilon, percentage in epsilon_array:
        y = openfile('./loss/lossfile_fed_poison_{}_cnn_100_iidFalse_dp_Gaussian_epsilon_1.0.dat'.format(epsilon))
        plt.plot(range(100), y, label='num_poison = {}, {}%'.format(epsilon, percentage))
    y = openfile('./loss/lossfile_fed_mnist_0_cnn_100_iidFalse_dp_Gaussian_epsilon_1.0.dat')
    plt.plot(range(100), y, label='num_poison = 0')
    plt.title('Mnist Gaussian')
    plt.legend()
    plt.savefig('./loss/mnist_gaussian_loss_eps_1.0.png')