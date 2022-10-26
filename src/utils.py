#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset

from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
from sampling import Dirichlet_noniid
from sampling import shakespeare,sent140

import numpy as np
from numpy.random import RandomState
# from random import Random
import random
from scipy import io

from models import MLP, NaiveCNN, BNCNN, ResNet, RNN, FeatureMLP

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def get_dataset(args,seed=None):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    rs = RandomState(seed)
    if args.dataset == 'cifar':
        train_dataset, test_dataset, user_groups, user_groups_test = get_cifar10(args, rs)
    elif args.dataset == 'cifar100':
        train_dataset, test_dataset, user_groups, user_groups_test = get_cifar100(args, rs)
    elif args.dataset == 'mnist' or args.dataset == 'fmnist':
        train_dataset, test_dataset, user_groups, user_groups_test = get_fmnist(args, rs)
    elif args.dataset == 'shake':
        args.num_classes = 80
        data_dir = '~/datasets/shakespeare/'
        user_groups_test={}
        train_dataset,test_dataset,user_groups=shakespeare(data_dir,args.shards_per_client,rs)
    elif args.dataset == 'sent':
        args.num_classes = 2
        data_dir = '~/datasets/sent140/'
        user_groups_test={}
        train_dataset,test_dataset,user_groups=sent140(data_dir,args.shards_per_client,rs)
    elif args.dataset == 'cifar10feature':
        args.num_classes = 10
        train_dataset, test_dataset, user_groups, user_groups_test = get_cifar10_feature(args, rs)
    else:
        raise RuntimeError("Not registered dataset! Please register it in utils.py")
    
    args.num_users=len(user_groups.keys())
    weights = []
    for i in range(args.num_users):
        weights.append(len(user_groups[i])/len(train_dataset))
    
    
    return train_dataset, test_dataset, user_groups, user_groups_test, np.array(weights)


def average_weights(w,omega=None):
    """
    Returns the average of the weights.
    """
    if omega is None:
        # default : all weights are equal
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
        return w_avg
        #omega = np.ones(len(w))
    omega = omega/np.sum(omega)
    w_avg = copy.deepcopy(w[0])
    for key in w[0].keys():
        avg_molecule = 0
        for i in range(len(w)):
            avg_molecule+=w[i][key]*omega[i]
        w_avg[key] = copy.deepcopy(avg_molecule)
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print('    Model     : {}'.format(args.model))
    print('    Optimizer : {}'.format(args.optimizer))
    print('    Learning  : {}'.format(args.lr))
    print('    Global Rounds   : {}'.format(args.epochs))

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')

    if args.afl:
        select_algo = "afl"
    elif args.power_d:
        select_algo = "powd"
    elif args.badge:
        select_algo = "badge"
    elif args.gpr_selection:
        select_algo = "gpr"
    else:
        select_algo = "random"
    print('    Client Selection   : {}'.format(select_algo))
    print('    Fraction of users  : {}'.format(args.frac))
    print('    Local Batch size   : {}'.format(args.local_bs))
    print('    Local Epochs       : {}\n'.format(args.local_ep))
    if args.FedProx:
        print('    Algorithm    :    FedProx({})'.format(args.mu))
    else:
        print('    Algorithm    :    FedAvg')
    return


def get_device(args):
    device = 'cuda:'+ args.gpu if args.gpu else 'cpu'
    if args.gpu:
        torch.cuda.set_device(device)
    return device


def get_model(args, data_size):
    if args.model == 'cnn':
        # Naive Convolutional neural netork
        global_model = NaiveCNN(args=args, 
            input_shape = data_size,
            final_pool=False)
    elif args.model == 'bncnn':
        # Convolutional neural network with batch normalization
        global_model = BNCNN(args = args, 
            input_shape = data_size)
    elif args.model == 'mlp' or args.model == 'log':
        # Multi-layer preceptron
        len_in = 1
        for x in data_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, 
                dim_hidden=args.mlp_layers if args.model=='mlp' else [],
                dim_out=args.num_classes)
    elif args.model == 'resnet':
        global_model = ResNet(args.depth,args.num_classes)
    elif args.model == 'rnn':
        if args.dataset=='shake':
            global_model = RNN(256,args.num_classes)
        else:
            # emb_arr,_,_= get_word_emb_arr('./data/sent140/embs.json')
            global_model = RNN(256,args.num_classes,300,True,128)
    elif args.model == 'featuremlp':
        len_in = 1
        for x in data_size:
            len_in *= x
            global_model = FeatureMLP(dim_in=len_in, 
                dim_hidden=args.mlp_layers if args.model=='featuremlp' else [],
                dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    return global_model


def get_filename(args):
    if not args.iid:
        base_file = './save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_{}[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.frac, args.iid,
                    'sp' if args.alpha is None else 'alpha',args.shards_per_client if args.alpha is None else args.alpha,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    else:
        base_file = './save/objects/{}_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_mu[{}]_lr[{:.5f}]'.\
                    format(args.dataset,'FedProx[%.3f]'%args.mu if args.FedProx else 'FedAvg', args.model, args.epochs,args.frac, args.iid,
                    args.local_ep, args.local_bs,args.mu,args.lr)
    if not os.path.exists(base_file):
        os.makedirs(base_file)

    if args.afl:
        file_name = base_file+'/afl'
    elif args.power_d:
        file_name = base_file+'/powerd_d[{}]'.format(args.d)
    elif args.badge:
        file_name = base_file+'/badge'
    elif not args.gpr_selection:
        file_name = base_file+'/random'
    else:
        file_name = base_file+'/gpr[int{}_gp{}_norm{}]_{}[{}]'.\
            format(args.GPR_interval,args.group_size,args.poly_norm,
            args.discount_method,args.loss_power if args.discount_method=='loss' else args.discount)

    return file_name


def get_cifar10(args, rs):
    args.num_classes = 10
    data_dir = '~/datasets/cifar10'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=apply_transform)

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, args.num_users,rs)
        user_groups_test = cifar_iid(test_dataset,args.num_users,rs)
    else:
        # Sample Non-IID user data from Mnist
        if args.alpha is not None:
            user_groups,_ = Dirichlet_noniid(train_dataset, args.num_users, args.num_classes, args.alpha, rs)
            user_groups_test,_ = Dirichlet_noniid(test_dataset, args.num_users, args.num_classes, args.alpha, rs)
        elif args.unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid(train_dataset, args.num_users,args.shards_per_client,rs)
            user_groups_test = cifar_noniid(test_dataset, args.num_users,args.shards_per_client,rs)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_cifar10_feature(args, rs):
    args.num_classes = 10
    data_dir = '/home/mlv/datasets/DataCIFAR10/' # FIXIT
    trn_name = 'CIFAR10Trn.mat'
    mat_path = os.path.join(data_dir, trn_name)
    mat = io.loadmat(mat_path)

    trnX = torch.Tensor(mat['Trn'][0][0][0])
    trnY = torch.Tensor(mat['Trn'][0][0][1]).argmax(axis=1)

    print(trnX.shape, trnY.shape)
    tst_name = 'CIFAR10Tst.mat'
    mat_path = os.path.join(data_dir, tst_name)
    mat = io.loadmat(mat_path)

    tstX = torch.Tensor(mat['Tst'][0][0][0])
    tstY = torch.Tensor(mat['Tst'][0][0][1]).argmax(axis=1)

    train_dataset = TensorDataset(trnX, trnY)
    test_dataset = TensorDataset(tstX, tstY)
    
    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, args.num_users,rs)
        user_groups_test = cifar_iid(test_dataset,args.num_users,rs)
    else:
        # Sample Non-IID user data from Mnist
        if args.alpha is not None:
            user_groups,_ = Dirichlet_noniid(train_dataset, args.num_users, args.num_classes, args.alpha, rs)
            user_groups_test,_ = Dirichlet_noniid(test_dataset, args.num_users, args.num_classes, args.alpha, rs)
        elif args.unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid(train_dataset, args.num_users,args.shards_per_client,rs)
            user_groups_test = cifar_noniid(test_dataset, args.num_users,args.shards_per_client,rs)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_cifar100(args, rs):
    args.num_classes = 100
    data_dir = '~/datasets/cifar100'
    apply_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                    transform=apply_transform)

    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                    transform=apply_transform)

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = cifar_iid(train_dataset, args.num_users,rs)
        user_groups_test = cifar_iid(test_dataset,args.num_users,rs)
    else:
        # Sample Non-IID user data from Mnist
        if args.alpha is not None:
            user_groups,_ = Dirichlet_noniid(train_dataset, args.num_users, args.num_classes, args.alpha, rs)
            user_groups_test,_ = Dirichlet_noniid(test_dataset, args.num_users, args.num_classes, args.alpha,rs)
        elif args.unequal:
            # Chose uneuqal splits for every user
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = cifar_noniid(train_dataset, args.num_users,args.shards_per_client,rs)
            user_groups_test = cifar_noniid(test_dataset, args.num_users,args.shards_per_client,rs)
    return train_dataset, test_dataset, user_groups, user_groups_test


def get_fmnist(args, rs):
    args.num_classes = 10
    data_dir = '~/datasets/mnist'

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                    transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                    transform=apply_transform)
    else:
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                    transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                    transform=apply_transform)

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = mnist_iid(train_dataset, args.num_users,rs)
        user_groups_test = mnist_iid(test_dataset,args.num_users,rs)
    else:
        # Sample Non-IID user data from Mnist
        if args.alpha is not None:
            user_groups,_ = Dirichlet_noniid(train_dataset, args.num_users, args.num_classes, args.alpha, rs)
            user_groups_test,_ = Dirichlet_noniid(test_dataset, args.num_users, args.num_classes, args.alpha,rs)
        elif args.unequal:
            # Chose uneuqal splits for every user
            user_groups = mnist_noniid_unequal(train_dataset, args.num_users,rs)
            user_groups_test = mnist_noniid_unequal(test_dataset, args.num_users,rs)
        else:
            user_groups = mnist_noniid(train_dataset, args.num_users,args.shards_per_client,rs)
            user_groups_test = mnist_noniid(test_dataset,args.num_users,args.shards_per_client,rs)

    return train_dataset, test_dataset, user_groups, user_groups_test


def get_last_param(model):
    """
    Get last parameter of model
    """
    for name in model.keys():
        param = model[name]
        if name[-7:] == '.weight':
            last_weight = copy.deepcopy(param).detach().cpu()
        elif name[-5:] == '.bias':
            last_bias = copy.deepcopy(param).unsqueeze(1).detach().cpu()

    last_param = torch.cat([last_weight, last_bias], dim=1)
    last_param = torch.flatten(last_param)
    return last_param


if __name__ == "__main__":
    from options import args_parser
    import matplotlib.pyplot as plt
    ALL_LETTERS = np.array(list("\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"))
    args = args_parser()
    args.dataset = 'sent'
    args.shards_per_client=1
    print(args.dataset)
    train_dataset, test_dataset, user_groups, user_groups_test,weights = get_dataset(args)
    print(len(train_dataset))
    print(len(test_dataset))
    # print(train_dataset[100][0].max())
    # print(''.join(ALL_LETTERS[train_dataset[0][0].numpy()].tolist()))
    # print(''.join(ALL_LETTERS[train_dataset[0][1].numpy()].tolist()))
    print(args.num_users)
    plt.hist(weights,bins=20)
    plt.show()
    
