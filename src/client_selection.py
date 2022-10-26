import copy
from math import ceil

import numpy as np
import torch
import torch.nn as nn

from update import train_federated_learning
from GPR import TrainGPR

def select_afl(args, m, AFL_Valuation):
    delete_num = int(args.alpha1*args.num_users)
    sel_num = int((1-args.alpha3)*m)
    tmp_value = np.vstack([np.arange(args.num_users), AFL_Valuation])
    tmp_value = tmp_value[:,tmp_value[1,:].argsort()]
    prob = np.exp(args.alpha2*tmp_value[1,delete_num:])
    prob = prob/np.sum(prob)
    sel1 = np.random.choice(np.array(tmp_value[0,delete_num:],dtype=np.int64),sel_num,replace=False,p=prob)
    remain = set(np.arange(args.num_users))-set(sel1)
    sel2 = np.random.choice(list(remain),m-sel_num,replace = False)
    idxs_users = np.append(sel1,sel2)

    return idxs_users


def select_powd(args, m, weights, gt_global_losses):
    A = np.random.choice(range(args.num_users), args.d, replace=False,p=weights)
    idxs_users = A[np.argsort(np.array(gt_global_losses[-1])[A])[-m:]]

    return idxs_users


def select_random(args, m):
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)

    return idxs_users


def gpr_test_offpolicy(args, global_model, weights,
    epoch, gpr_idxs_users, train_dataset, user_groups, gt_global_losses,
    gpr, offpolicy_losses, gpr_loss_decrease, gpr_acc_improve, train_accuracy):
    if args.verbose:
        print("Training with GPR Selection:")
    gpr_acc,gpr_loss = train_federated_learning(args,epoch,
                        copy.deepcopy(global_model),gpr_idxs_users,train_dataset,user_groups)
    gpr_loss_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gpr_loss)-np.array(gt_global_losses[-1]),1),
                                    np.ones([args.num_users,1])],1)
    predict_loss,_,_=gpr.Predict_Loss(gpr_loss_data,gpr_idxs_users,np.delete(list(range(args.num_users)),gpr_idxs_users))
    print("GPR Predict Off-Policy Loss:{:.4f}".format(predict_loss))
    offpolicy_losses.append(predict_loss)

    gpr_dloss = np.sum((np.array(gpr_loss)-np.array(gt_global_losses[-1]))*weights)
    gpr_loss_decrease.append(gpr_dloss)
    gpr_acc_improve.append(gpr_acc-train_accuracy[-1])
    if args.verbose:
        print("Training with {} Selection".format('Random' if not args.power_d else 'Power-D'))


def gpr_warmup(args, epoch, gpr,
    gt_global_losses, gpr_data):
    gpr.update_loss(np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gt_global_losses[-1]),1)],1))
    epoch_gpr_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),1),
                                    np.ones([args.num_users,1])],1)
    gpr_data.append(epoch_gpr_data)
    print("Training GPR")
    TrainGPR(gpr,gpr_data[max([(epoch-args.gpr_begin-args.group_size+1),0]):epoch-args.gpr_begin+1],
            args.train_method,lr = 1e-2,llr = 0.0,gamma = args.GPR_gamma,max_epoches=args.GPR_Epoch+50,schedule_lr=False,verbose=args.verbose)


def gpr_optimal(args, epoch, gpr, m, 
    global_model, train_dataset, user_groups,
    gt_global_losses, gpr_data):
    gpr.update_loss(np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                                np.expand_dims(np.array(gt_global_losses[-1]),1)],1))
    gpr.Reset_Discount()
    print("Training with Random Selection For GPR Training:")
    random_idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    gpr_acc,gpr_loss = train_federated_learning(args,epoch,
                        copy.deepcopy(global_model),random_idxs_users,train_dataset,user_groups)
    epoch_gpr_data = np.concatenate([np.expand_dims(list(range(args.num_users)),1),
                                    np.expand_dims(np.array(gpr_loss)-np.array(gt_global_losses[-1]),1),
                                    np.ones([args.num_users,1])],1)
    gpr_data.append(epoch_gpr_data)
    print("Training GPR")
    TrainGPR(gpr,gpr_data[-ceil(args.group_size/args.GPR_interval):],
            args.train_method,lr = 1e-2,llr = 0.0,gamma = args.GPR_gamma**args.GPR_interval,max_epoches=args.GPR_Epoch,schedule_lr=False,verbose=args.verbose)


def select_badge(args, m, prev_params):
    params = torch.stack(prev_params)
    pdist_mat = torch.full((m, args.num_users), torch.inf)
    selected_clients = []

    # using k-means++
    seed_client = np.random.choice(args.num_users, 
        1, replace=False).item()
    selected_clients.append(seed_client)
    pdist_mat[0:,] = get_pairdistance(params[seed_client,:], params)

    for t in range(1, m):
        pmf, _ = pdist_mat.min(dim=0)
        pmf = torch.div(pmf, pmf.sum())

        select = pmf.multinomial(num_samples=1, replacement=False).item()
        selected_clients.append(select)

        pdist_mat[t,:] = get_pairdistance(params[select,:], params)
    
    return selected_clients


def get_pairdistance(vec1, vec2):
    """
    """
    pdist = nn.PairwiseDistance(p=2)
    ret = pdist(vec1, vec2)

    return ret