#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.multiprocessing

# from language_utils import get_word_emb_arr

from options import args_parser
from update import LocalUpdate,test_inference, federated_test_idx

from utils import (get_dataset, 
    average_weights, 
    exp_details, 
    setup_seed, 
    get_model, get_device, get_filename)

from mvnt import MVN_Test
from logger import get_writter, log_experiment
import GPR
from GPR import Kernel_GPR
from client_selection import (select_afl, select_powd, select_random, 
    gpr_test_offpolicy, gpr_warmup, gpr_optimal)


def oneseed_experiment(args, seed, file_name):
    # initialization
    writer = get_writter(args, seed)
    device = get_device(args)
    start_time = time.time()

    print("Start with Random Seed: {}".format(seed))

    # load dataset and user groups
    train_dataset, test_dataset, user_groups, _, weights = get_dataset(args,seed)
    # weights /=np.sum(weights)
    if seed is not None:
        setup_seed(seed)
    
    # BUILD MODEL
    data_size = train_dataset[0][0].shape
    global_model = get_model(args, data_size)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Build GP
    if args.gpr:
        if args.kernel=='Poly':
            gpr = Kernel_GPR(args.num_users,dimension = args.dimension,init_noise=0.01,
                                order = 1,Normalize = args.poly_norm,kernel=GPR.Poly_Kernel,loss_type= args.train_method)
        elif args.kernel=='SE':
            gpr = Kernel_GPR(args.num_users,dimension = args.dimension,init_noise=0.01,kernel=GPR.SE_Kernel,loss_type= args.train_method)
        else:
            gpr = GPR.Matrix_GPR(args.num_users,loss_type=args.train_method)
        # gpr.to(device)

    # copy weights
    global_weights = global_model.state_dict()
    local_weights = []# store local weights of all users for averaging
    local_states = []# store local states of all users, these parameters should not be uploaded
    
    for i in range(args.num_users):
        local_states.append(copy.deepcopy(global_model.Get_Local_State_Dict()))
        local_weights.append(copy.deepcopy(global_weights))

    local_states = np.array(local_states)
    local_weights = np.array(local_weights)

    # Training
    train_loss, train_accuracy = [], []
    test_loss,test_accuracy = [],[]

    loss_prev = 0.0
    local_losses = []# test losses evaluated on local models(before averaging)
    # global_losses = []# test losses evaluated on global models(after averaging)
    chosen_clients = []# chosen clients on each epoch
    gt_global_losses = []# test losses on global models(after averaging) over all clients
    gpr_data = []# GPR Training data
    print_every = 1
    init_mu = args.mu

    gpr_idxs_users = None
    gpr_loss_decrease = []
    gpr_acc_improve = []
    rand_loss_decrease = []
    rand_acc_improve = []

    predict_losses = []
    offpolicy_losses = []
    # mu = []
    sigma = []
    sigma_gt=[]

    # Test the global model before training
    list_acc, list_loss = federated_test_idx(args,global_model,
                                            list(range(args.num_users)),
                                            train_dataset,user_groups)
    gt_global_losses.append(list_loss)
    train_accuracy.append(sum(list_acc)/len(list_acc))
    
    if args.afl:
        AFL_Valuation = np.array(list_loss)*np.sqrt(weights*len(train_dataset))

    # gpr_loss_data = None
    for epoch in tqdm(range(args.epochs)):
        print('\n | Global Training Round : {} |\n'.format(epoch+1))
        epoch_global_losses = []
        epoch_local_losses = []
        global_model.train()
        if args.dataset=='cifar' or epoch in args.schedule:
            args.lr*=args.lr_decay

        if gpr_idxs_users is not None and not args.gpr_selection:
            gpr_test_offpolicy(args, global_model, weights,
                epoch, gpr_idxs_users, train_dataset, user_groups, gt_global_losses,
                gpr, offpolicy_losses, gpr_loss_decrease, gpr_acc_improve, train_accuracy)
        
        m = max(int(args.frac * args.num_users), 1)

        if args.afl:
            idxs_users = select_afl(args, m, AFL_Valuation)
        elif args.power_d:
            idxs_users = select_powd(args, m, weights, gt_global_losses)
        elif not args.gpr_selection or gpr_idxs_users is None:
            idxs_users = select_random(args, m)
        else:
            # FedGP
            idxs_users = copy.deepcopy(gpr_idxs_users)

        chosen_clients.append(idxs_users)
        
        for idx in idxs_users:
            local_model = copy.deepcopy(global_model)
            local_update = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx] ,global_round = epoch)
            w,test_loss,init_test_loss = local_update.update_weights(model=local_model)
            
            local_states[idx] = copy.deepcopy(local_model.Get_Local_State_Dict())
            local_weights[idx]=copy.deepcopy(w)
            epoch_global_losses.append(init_test_loss)# TAKE CARE: this is the test loss evaluated on the (t-1)-th global weights!
            epoch_local_losses.append(test_loss)

        # update global weights
        if args.global_average:
            global_weights = average_weights(local_weights,omega=None)
        else:
            global_weights = average_weights(local_weights[idxs_users],omega=None)

        for i in range(args.num_users):
            local_weights[i] = copy.deepcopy(global_weights)
        # update global weights
        global_model.load_state_dict(global_weights)

        if args.afl:
            AFL_Valuation[idxs_users] = np.array(epoch_global_losses)*np.sqrt(weights[idxs_users]*len(train_dataset))
        # global_losses.append(epoch_global_losses)
        local_losses.append(epoch_local_losses)

        # dynamic mu for FedProx
        loss_avg = sum(epoch_local_losses) / len(epoch_local_losses)
        if args.dynamic_mu and epoch > 0:
            if loss_avg > loss_prev:
                args.mu+=init_mu*0.1
            else:
                args.mu=max([args.mu-init_mu*0.1,0.0])
        loss_prev = loss_avg
        train_loss.append(loss_avg)

        # calculate test accuracy over all users
        list_acc, list_loss = federated_test_idx(args,global_model,
                                                list(range(args.num_users)),
                                                train_dataset,user_groups)
        gt_global_losses.append(list_loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # calculate the advantage in off-policy
        if gpr_idxs_users is not None and not args.gpr_selection:
            rand_loss_decrease.append(np.sum((np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]))*weights))
            rand_acc_improve.append(train_accuracy[-1]-train_accuracy[-2])
            print("Advantage:",gpr_loss_decrease[-1]-rand_loss_decrease[-1])
        
        # test prediction accuracy of GP model
        if args.gpr and epoch>args.warmup:
            test_idx = np.random.choice(range(args.num_users), m, replace=False)
            test_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                        np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                        np.ones([args.num_users,1])],1)
            pred_idx = np.delete(list(range(args.num_users)),test_idx)
            
            predict_loss,mu_p,sigma_p = gpr.Predict_Loss(test_data,test_idx,pred_idx)
            print("GPR Predict relative Loss:{:.4f}".format(predict_loss))
            predict_losses.append(predict_loss)

        # train and exploit GPR
        if args.gpr:
            if epoch<=args.warmup and epoch>=args.gpr_begin:# warm-up
                gpr_warmup(args, epoch, gpr, gt_global_losses, gpr_data)
            elif epoch>args.warmup and epoch%args.GPR_interval==0:# normal and optimization round
                gpr_optimal(args, epoch, gpr, m, global_model, train_dataset, user_groups, gt_global_losses, gpr_data) 
            else:# normal and not optimization round
                gpr.update_loss(np.concatenate([np.expand_dims(idxs_users,1),
                                            np.expand_dims(epoch_global_losses,1)],1))
                gpr.update_discount(idxs_users,args.discount)
                
            if epoch>=args.warmup:
                gpr_idxs_users = gpr.Select_Clients(m,args.loss_power,args.epsilon_greedy,args.discount_method,weights,args.dynamic_C,args.dynamic_TH)
                print("GPR Chosen Clients:",gpr_idxs_users)
            
            if args.mvnt and (epoch==args.warmup or (epoch%args.mvnt_interval==0 and epoch>args.warmup)):
                mvn_file = file_name+'/MVN/{}'.format(seed)
                if not os.path.exists(mvn_file):
                    os.makedirs(mvn_file)
                mvn_samples=MVN_Test(args,copy.deepcopy(global_model),
                                            train_dataset,user_groups,
                                            file_name+'/MVN/{}/{}.csv'.format(seed,epoch))
                sigma_gt.append(np.cov(mvn_samples,rowvar=False,bias = True))
                sigma.append(gpr.Covariance().clone().detach().numpy())

        # test inference on the global test dataset
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        test_accuracy.append(test_acc)
        if args.target_accuracy is not None:
            if test_acc>=args.target_accuracy:
                break

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            training_loss = np.sum(np.array(list_loss)*weights)
            print(' \nAvg Training Stats after {} global rounds:'.format(epoch+1))
            print('Training Loss : {}'.format(training_loss))
            print('Train Accuracy: {:.2f}%'.format(100*train_accuracy[-1]))
            print("Test Accuracy: {:.2f}%\n".format(100*test_acc))
            log_experiment(writer, training_loss, 100*train_accuracy[-1], 100*test_acc, epoch)
    
    print(' \n Results after {} global rounds of training:'.format(epoch+1))
    print("|---- Final Test Accuracy: {:.2f}%".format(100*test_accuracy[-1]))
    # print("|---- Max Train Accuracy: {:.2f}%".format(100*max(train_accuracy)))
    print("|---- Max Test Accuracy: {:.2f}%".format(100*max(test_accuracy)))

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # save the training records:
    with open(file_name+'_{}.pkl'.format(seed), 'wb') as f:
        pickle.dump([train_loss, train_accuracy,chosen_clients,
                    weights,None if not args.gpr else gpr.state_dict(),
                    gt_global_losses,test_accuracy], f)
    
    if args.mvnt:
        with open(file_name+'/MVN/{}/Sigma.pkl'.format(seed), 'wb') as f:
            pickle.dump([sigma,sigma_gt],f)


def main(args):
    os.environ["OUTDATED_IGNORE"]='1'
    torch.multiprocessing.set_start_method('spawn')

    gargs = copy.deepcopy(args)
    exp_details(args)
    
    file_name = get_filename(args)
    
    if gargs.seed is None or gargs.iid:
        gargs.seed = [None,]
    for seed in gargs.seed:
        args = copy.deepcopy(gargs) # recover the args
        oneseed_experiment(args, seed, file_name)


if __name__ == '__main__':
    args = args_parser()
    main(args)
