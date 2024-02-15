#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import numpy as np


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available() and args.gpu != -1:
            data, target = data.cuda(args.device), target.cuda(args.device)
        else:
            data, target = data.cpu(), target.cpu()
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print("I got ", correct, " from these many ", len(data_loader.dataset), " examples.")
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss


def test_img_poison(net_g, data, target, args, numpoison, numepoch):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    # data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data)
    iterations = int(math.ceil(l / 200))

    true_true = 0
    false_true = 0
    true_false = 0
    false_false = 0

    low_idx = 0
    high_idx = 200

    for i in range(iterations):
        data_cut, target_cut = data[low_idx:high_idx], target[low_idx:high_idx]
        data_numerical = data_cut.numpy()
        data_numerical = data_numerical.astype('float32')
        data_numerical /= 255
        data_numerical = np.reshape(data_numerical, (len(data_numerical), 1, 28, 28))
        data_cut = torch.from_numpy(data_numerical)

        data_cut = (data_cut - 0.1307) / 0.3081

        if torch.cuda.is_available() and args.gpu != -1:
            data_cut, target_cut = data_cut.cuda(args.device), target_cut.cuda(args.device)
        else:
            data_cut, target_cut = data_cut.cpu(), target_cut.cpu()


        log_probs = net_g(data_cut)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target_cut, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        current_correct = y_pred.eq(target_cut.data.view_as(y_pred)).long().cpu().sum()
        correct += current_correct

        # for (one_img, one_label) in zip(data_cut, target_cut.cpu().numpy()):
        #     one_label_array = []
        #     one_label_array.append(one_label)
        #     one_label = torch.from_numpy(np.array(one_label_array))
        #     one_label = one_label.cuda(args.device)
        #     print("hello")

        if numepoch == 99:
            batch_loss = F.cross_entropy(log_probs, target_cut, reduction='sum').item()
            batch_acc = 100.00 * current_correct / len(data_cut)
            original_acc = 17.73
            original_loss = 2.3
            threshold = 0.03
            if args.dp_epsilon <= 30:
                if low_idx >= l - numpoison:
                    for (one_img, one_label) in zip(data_cut, target_cut.cpu().numpy()):
                        one_img = one_img.cpu()
                        one_img.numpy()
                        one_img = np.reshape(one_img, (1, 1, 28, 28))
                        one_img = one_img.cuda(args.device)
                        probs = net_g(one_img)
                        # sum up batch loss

                        one_label_array = []
                        one_label_array.append(one_label)
                        one_label = torch.from_numpy(np.array(one_label_array))
                        one_label = one_label.cuda(args.device)

                        indv_loss = F.cross_entropy(probs, one_label, reduction='sum').item()
                        if indv_loss >= original_loss + threshold:
                            true_true += 1
                        else:
                            false_false += 1
                    # Here all the rest are poisoned
                elif high_idx >= l - numpoison:
                    # data_poison = data[l - numpoison:]
                    relative_pos = low_idx
                    for (one_img, one_label) in zip(data_cut, target_cut.cpu().numpy()):
                        one_img = one_img.cpu()
                        one_img.numpy()
                        one_img = np.reshape(one_img, (1, 1, 28, 28))
                        one_img = one_img.cuda(args.device)
                        probs = net_g(one_img)
                        # sum up batch loss

                        one_label_array = []
                        one_label_array.append(one_label)
                        one_label = torch.from_numpy(np.array(one_label_array))
                        one_label = one_label.cuda(args.device)


                        indv_loss = F.cross_entropy(probs, one_label, reduction='sum').item()
                        if relative_pos >= l - numpoison and indv_loss >= original_loss + threshold:
                            true_true += 1
                        elif relative_pos >= l - numpoison:
                            false_false += 1
                        elif relative_pos < l - numpoison and indv_loss >= original_loss + threshold:
                            false_true += 1
                        else:
                            true_false += 1
                else:
                    for (one_img, one_label) in zip(data_cut, target_cut.cpu().numpy()):
                        one_img = one_img.cpu()
                        one_img.numpy()
                        one_img = np.reshape(one_img, (1, 1, 28, 28))
                        one_img = one_img.cuda(args.device)
                        probs = net_g(one_img)
                        # sum up batch loss

                        one_label_array = []
                        one_label_array.append(one_label)
                        one_label = torch.from_numpy(np.array(one_label_array))
                        one_label = one_label.cuda(args.device)

                        indv_loss = F.cross_entropy(probs, one_label, reduction='sum').item()
                        if indv_loss >= original_loss + threshold:
                            false_true += 1
                        else:
                            true_false += 1
            else:
                if low_idx >= l - numpoison:
                    # Here all the rest are poisoned
                    if batch_acc <= original_acc - 0.2:
                        true_true += len(data_cut)
                    else:
                        false_false += len(data_cut)
                elif high_idx >= l - numpoison:
                    # data_poison = data[l - numpoison:]
                    if batch_acc <= original_acc - 0.2:
                        true_true += len(data_cut)
                    else:
                        false_false += len(data_cut)
                else:
                    if batch_acc <= original_acc - 0.2:
                        false_true += len(data_cut)
                    else:
                        true_false += len(data_cut)

        low_idx += 200
        high_idx += 200
    confusion_matrix = [[true_true, false_true], [false_false, true_false]]
    test_loss /= len(data)
    print("I got ", correct.item(), " from ", len(data), " examples.")
    accuracy = 100.00 * correct / len(data)
    true_true = 100.00 * true_true / len(data)
    return accuracy, test_loss, confusion_matrix
