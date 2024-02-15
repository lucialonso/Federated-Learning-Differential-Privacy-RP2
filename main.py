#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time

from torch.utils.data import DataLoader

import poison

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import seaborn

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from models.test import test_img_poison
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule

if __name__ == '__main__':
    # parse args

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("Epsilon ============== ", args.dp_epsilon)
    epsilon_array = [500, 1500, 3000, 5000]
    for pois_num in epsilon_array:
        dict_users = {}
        dataset_train, dataset_test = None, None
        poisoned_labels = None
        numImages = pois_num

        # load dataset and split users
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
            args.num_channels = 1
            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.dataset == 'poison':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
            args.num_channels = 1

            poisoned_data, poisoned_labels = poison.poison_images(numImages)
            # poisoned_data = poisoned_data.astype('float32')
            # poisoned_data /= 255
            # poisoned_data = np.reshape(poisoned_data, (len(poisoned_data), 28, 28))
            #
            # mean = poisoned_data.mean()
            # std = poisoned_data.std()
            # poisoned_data = (poisoned_data-mean) / std
            poisoned_data = torch.from_numpy(poisoned_data)

            data_loader = DataLoader(dataset_test)

            poisoned_labels = torch.from_numpy(poisoned_labels)
            dataset_test_labels = data_loader.dataset.targets
            poisoned_labels = torch.cat((dataset_test_labels, poisoned_labels))

            dataset_test = data_loader.dataset.data
            dataset_test = torch.cat((dataset_test, poisoned_data))

            # sample users
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.dataset == 'cifar':
            # trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            args.num_channels = 3
            trans_cifar_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            trans_cifar_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
            dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
            if args.iid:
                dict_users = cifar_iid(dataset_train, args.num_users)
            else:
                dict_users = cifar_noniid(dataset_train, args.num_users)
        elif args.dataset == 'fashion-mnist':
            args.num_channels = 1
            trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
            dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                                  transform=trans_fashion_mnist)
            dataset_test = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                                 transform=trans_fashion_mnist)
            if args.iid:
                dict_users = mnist_iid(dataset_train, args.num_users)
            else:
                dict_users = mnist_noniid(dataset_train, args.num_users)
        elif args.dataset == 'femnist':
            args.num_channels = 1
            dataset_train = FEMNIST(train=True)
            dataset_test = FEMNIST(train=False)
            dict_users = dataset_train.get_client_dic()
            args.num_users = len(dict_users)
            if args.iid:
                exit('Error: femnist dataset is naturally non-iid')
            else:
                print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
        elif args.dataset == 'shakespeare':
            dataset_train = ShakeSpeare(train=True)
            dataset_test = ShakeSpeare(train=False)
            dict_users = dataset_train.get_client_dic()
            args.num_users = len(dict_users)
            if args.iid:
                exit('Error: ShakeSpeare dataset is naturally non-iid')
            else:
                print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
        else:
            exit('Error: unrecognized dataset')
        img_size = dataset_train[0][0].shape

        net_glob = None
        # build model
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.model == 'cnn' and args.dataset == 'poison':
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.dataset == 'femnist' and args.model == 'cnn':
            net_glob = CNNFemnist(args=args).to(args.device)
        elif args.dataset == 'shakespeare' and args.model == 'lstm':
            net_glob = CharLSTM().to(args.device)
        elif args.model == 'mlp':
            len_in = 1
            for x in img_size:
                len_in *= x
            net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        # use opacus to wrap model to clip per sample gradient
        if args.dp_mechanism != 'no_dp':
            net_glob = GradSampleModule(net_glob)
        print(net_glob)
        net_glob.train()

        # copy weights
        w_glob = net_glob.state_dict()
        all_clients = list(range(args.num_users))

        # training
        acc_test = []
        loss_test = []
        if args.serial:
            clients = [LocalUpdateDPSerial(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in
                       range(args.num_users)]
        else:
            clients = [LocalUpdateDP(args=args, dataset=dataset_train, idxs=dict_users[i]) for i in range(args.num_users)]
        m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)
        for iter in range(args.epochs):
            t_start = time.time()
            w_locals, loss_locals, weight_locols = [], [], []
            # round-robin selection
            begin_index = (iter % loop_index) * m
            end_index = begin_index + m
            idxs_users = all_clients[begin_index:end_index]
            for idx in idxs_users:
                local = clients[idx]
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                loss_locals.append(copy.deepcopy(loss))
                weight_locols.append(len(dict_users[idx]))

            # update global weights
            w_glob = FedWeightAvg(w_locals, weight_locols)
            # copy weight to net_glob
            net_glob.load_state_dict(w_glob)

            # print accuracy
            net_glob.eval()
            if args.dataset != 'poison':
                acc_t, loss_t = test_img(net_glob, dataset_test, args)
            else:
                if iter == 99:
                    acc_t, loss_t, confusion_matrix = test_img_poison(net_glob, dataset_test, poisoned_labels, args,
                                                                      numImages, iter)
                else:
                    acc_t, loss_t, _ = test_img_poison(net_glob, dataset_test, poisoned_labels, args,
                                                                      numImages, iter)
            t_end = time.time()
            print(
                "Round {:3d}, Testing accuracy: {:.2f}, Time:  {:.2f}s, Loss:  {:.2f}".format(iter, acc_t, t_end - t_start,
                                                                                              loss_t))

            acc_test.append(acc_t.item())
            loss_test.append(np.float32(loss_t))

        rootpathconf = './conf'
        if not os.path.exists(rootpathconf):
            os.makedirs(rootpathconf)
        plt.figure()
        ax = seaborn.heatmap(confusion_matrix, xticklabels='PN', yticklabels='PN', annot=True, square=True, cmap='Blues', fmt=',d')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        plt.savefig(rootpathconf + '/correct-loss/heat_poison_{}_eps_{}_threshold_0.03.png'.format(numImages, args.dp_epsilon))



        # rootpathacc = './acc'
        # if not os.path.exists(rootpathacc):
        #     os.makedirs(rootpathacc)
        # accfile = open(rootpathacc + '/accfile_fed_{}_{}_{}_{}_iid{}_dp_{}_epsilon_{}.dat'.
        #                format(args.dataset, numImages, args.model, args.epochs, args.iid,
        #                       args.dp_mechanism, args.dp_epsilon), "w")
        #
        # for ac in acc_test:
        #     sac = str(ac)
        #     accfile.write(sac)
        #     accfile.write('\n')
        # accfile.close()
        #
        # rootpathloss = './loss'
        # if not os.path.exists(rootpathloss):
        #     os.makedirs(rootpathloss)
        # lossfile = open(rootpathloss + '/lossfile_fed_{}_{}_{}_{}_iid{}_dp_{}_epsilon_{}.dat'.
        #                 format(args.dataset, numImages, args.model, args.epochs, args.iid,
        #                        args.dp_mechanism, args.dp_epsilon), "w")
        #
        # for loss_x in loss_test:
        #     sloss_x = str(loss_x)
        #     lossfile.write(sloss_x)
        #     lossfile.write('\n')
        # lossfile.close()

        # # plot loss curve
        # plt.figure()
        # plt.plot(range(len(acc_test)), acc_test)
        # plt.ylabel('test accuracy')
        # plt.savefig(rootpathacc + '/fed_{}_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_acc.png'.format(
        #     args.dataset, numImages, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon))
        #
        # # plot loss curve
        # plt.figure()
        # plt.plot(range(len(loss_test)), loss_test)
        # plt.ylabel('test loss')
        # plt.savefig(rootpathloss + '/fed_{}_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_loss.png'.format(
        #     args.dataset, numImages, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon))
