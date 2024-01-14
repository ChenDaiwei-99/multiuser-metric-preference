import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import DataLoader, Dataset

# system related packages
from copy import deepcopy
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import sys
import re
import itertools

# customized packages
import group_dataset
import dataset
import utils
from mlp import projector
from metricpref_learner import MetricPrefLearner, MetricPrefLearner_groups

def train(learner,optimizer,loss_fn,train_loader,test_loader,epochs,relative_error_ind=True, true_M=None, true_u=None, groups=None):
    train_stats = {
        'train_loss_per_batch_record': [],
        'train_accu_record': [],
        'test_loss_per_batch_record': [],
        'test_accu_record': [],
        'relative_metric_error_record': [],
        'relative_ideal_point_error_record': [],
        'relative_weights_error_record': [],
    }
    tqdmr = tqdm(range(epochs))
    for ep in tqdmr:
        for x,y in train_loader:
            optimizer.zero_grad()
            pred_delta = learner(x)
            ############################ Evaluation part start #############################
            acc_batch = torch.sum((pred_delta * y)>0)
            if relative_error_ind:
                # calculate the relative_metric_error
                L = learner.model.layers.weight
                relative_metric_error = torch.norm(L.T @ L - torch.tensor(true_M)) / torch.norm(torch.tensor(true_M))
                train_stats['relative_metric_error_record'].append(relative_metric_error.item())
                # calculate the relative_ideal_point_error
                if groups:
                    us = learner.us_groups @ learner.softmax(learner.unconstrained_weights)
                    relative_ideal_point_error = (torch.norm(torch.tensor(true_u)-us) / torch.norm(torch.tensor(true_u))).item()
                    train_stats['relative_ideal_point_error_record'].append(relative_ideal_point_error)
                #     alpha = learner.softmax(learner.unconstrained_weights).T
                #     relative_ideal_point_error = (torch.norm(torch.tensor(true_u)-learner.us_groups) / torch.norm(torch.tensor(true_u))).item()
                #     train_stats['relative_ideal_point_error_record'].append(relative_ideal_point_error)
                #     relative_weights_error_record = (torch.norm(torch.tensor(true_alpha)-alpha) / torch.norm(torch.tensor(true_alpha))).item()
                #     train_stats['relative_weights_error_record'].append(relative_weights_error_record)
                else:
                    relative_ideal_point_error = (torch.norm(torch.tensor(true_u)-learner.us) / torch.norm(torch.tensor(true_u))).item()
                    train_stats['relative_ideal_point_error_record'].append(relative_ideal_point_error)
            ############################ Evaluation part end #############################   
            loss = loss_fn(pred_delta,y)
            train_stats['train_loss_per_batch_record'].append(loss.item())
            train_stats['train_accu_record'].append(acc_batch/len(y))
            loss.backward()
            optimizer.step()
        val_stat = val(learner,loss_fn,test_loader)
        train_stats['test_loss_per_batch_record'].extend(val_stat['test_loss_per_batch'])
        train_stats['test_accu_record'].append(val_stat['test_accu'])
        tqdmr.set_postfix({'test_accu': val_stat['test_accu']})
    return train_stats

def val(learner,loss_fn,test_loader):
    total_val_samples = len(test_loader.dataset)
    val_stat = {
        'test_correct': 0,
        'test_loss_per_batch': [],
        'test_accu': 0
    }
    with torch.no_grad():
        for x,y in test_loader:
            pred_delta = learner(x)
            acc_batch = torch.sum((pred_delta * y)>0)
            loss = loss_fn(pred_delta,y)
            val_stat['test_correct'] += acc_batch.item()
            val_stat['test_loss_per_batch'].append(loss.item())
    val_stat['test_accu'] = val_stat['test_correct'] / total_val_samples
    return val_stat

class CustomDataset(Dataset):   # written by the gpt-4 :)
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]
    
def hinge_loss(outputs, targets):   # written by the gpt-4 :)
    hinge_loss_value = 1 - targets * outputs
    hinge_loss_value = torch.clamp(hinge_loss_value, min=0)
    return hinge_loss_value.mean()

def logistic(z):
    return 1 / (1 + torch.exp(-z))

def nll_logistic(predictions, targets):
    # Ensure targets are -1 or 1
    assert torch.all((targets == -1) | (targets == 1))
    probabilities = logistic(targets * predictions)
    nll = -torch.sum(torch.log(probabilities))
    return nll

# in order to keep using the same dataset
# we only initialize the dataset for one time
def get_dataset(args):
    ########### hyperparameters for the dataset ################
    feature_dim = args['feature_dim']
    metric_rank = args['metric_rank']
    num_items = args['num_items']
    num_users = args['num_users']
    num_pairs_per_user = args['num_pairs_per_user']
    noise_type = args['noise_type']
    noise_beta = args['noise_beta']
    num_groups = args['num_groups']
    normal_dataset = group_dataset.Dataset(dataset_type='Normal', d=feature_dim, 
                                            r=metric_rank, n=num_items, 
                                            N=num_users, m=num_pairs_per_user, 
                                            noise_type=noise_type, 
                                            noise_param=noise_beta, X=None,
                                            num_groups=num_groups)
    return normal_dataset

def train_main(args, normal_dataset, relative_error_ind=True):
    ########### hyperparameters for the dataset ################
    feature_dim = args['feature_dim']
    metric_rank = args['metric_rank']
    num_items = args['num_items']
    num_users = args['num_users']
    samples_per_user = args['samples_per_user']
    num_groups = args['num_groups']
    set_groups = args['set_groups']
    ########### hyperparameters for the training ################
    # Notice: these hyperparameters are chosen by the grid_search experiments
    epochs = args['epochs'] #2000
    bs = args['bs']#64
    lr = args['lr']#0.005  # TODO: use different learning rate for us and net
    weight_decay_us = args['weight_decay_us']#0
    weight_decay_net = args['weight_decay_net']#0.001
    weight_decay_unconstrained_weight = args['weight_decay_unconstrained_weight']#0
    optimizer_name = args['optimizer_name']#'adam'
    ########### hyperparameters for the ablation experiments ################
    ablate_m = args['ablate_m']
    ablate_u = args['ablate_u']
    ablate_alpha = args['ablate_alpha']
    ########### initialize the dataloders ################
    normal_data = normal_dataset.getAllData()
    items, observations, true_y, true_M, true_u = normal_data['X'], normal_data['S'], normal_data['Y'], normal_data['M'], normal_data['U']
    obs_train, obs_test, Y_train, Y_test = normal_dataset.getTrainTestSplit(train_size=num_users*samples_per_user)
    ############ define dataloader ################
    train_dataset = CustomDataset(list(zip(obs_train, Y_train)))
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataset = CustomDataset(list(zip(obs_test, Y_test)))
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=True)
    ############ initialize the model and the learner################
    net = projector(feature_dim=feature_dim, num_layer=1, num_class=feature_dim, bias_ind=False)
    learner = MetricPrefLearner_groups(dim_feature=feature_dim, num_users=num_users, items=items, num_groups=set_groups)
    learner.assignModel(net)
    us_params = []
    net_params = []
    unconstrained_weights_params = []
    for name, param in learner.named_parameters():
        if 'us' in name:
            us_params.append(param)
        elif 'model' in name:
            net_params.append(param)
        elif 'unconstrained_weights' in name:
            if not ablate_alpha: unconstrained_weights_params.append(param)
    ############ define loss and optimizer ################
    if not ablate_alpha:
        weight_decay_dic = [
            {'params': us_params, 'weight_decay': weight_decay_us},
            {'params': net_params, 'weight_decay': weight_decay_net},
            {'params': unconstrained_weights_params, 'weight_decay': weight_decay_unconstrained_weight},
        ]
    else:
        weight_decay_dic = [
            {'params': us_params, 'weight_decay': weight_decay_us},
            {'params': net_params, 'weight_decay': weight_decay_net},
        ]
    loss_fn = hinge_loss
    if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(weight_decay_dic, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(weight_decay_dic, lr=lr, momentum=0.9)

    # important modification: record the metric error
    # if num_groups:
    #     if args['set_groups'] == num_groups:
    #         train_stats = train(learner, optimizer, loss_fn, train_dataloader, test_dataloader, epochs, 
    #                         relative_error_ind=relative_error_ind, true_M=true_M, true_u=true_u,groups=True,true_alpha=true_alpha)
    #     else:
    #         train_stats = train(learner, optimizer, loss_fn, train_dataloader, test_dataloader, epochs, 
    #                         relative_error_ind=False)
    # else:
    if num_groups:
        train_stats = train(learner, optimizer, loss_fn, train_dataloader, test_dataloader, epochs, 
                        relative_error_ind=relative_error_ind, true_M=true_M, true_u=true_u, groups=num_groups)
    else:
        train_stats = train(learner, optimizer, loss_fn, train_dataloader, test_dataloader, epochs, 
                        relative_error_ind=relative_error_ind, true_M=true_M, true_u=true_u)

    return train_stats, learner

def oracle_pred(normal_dataset):
    # Oracle prediction (just for verification) original version
    normal_data = normal_dataset.getAllData()

    items, observations, true_y, true_M, true_u = normal_data['X'], normal_data['S'], normal_data['Y'], normal_data['M'], normal_data['U']

    delta_s = []
    pred_ys = []
    for obs in observations:
        user_id, comparison_pair = obs
        x_i, x_j = comparison_pair
        delta = (items[:,x_i]-true_u[:,user_id]).T @ true_M @ (items[:,x_i]-true_u[:,user_id]) - (items[:,x_j]-true_u[:,user_id]).T @ true_M @ (items[:,x_j]-true_u[:,user_id])
        delta_s.append(delta)
        if delta > 0:
            pred_y = 1
        else:
            pred_y = -1
        pred_ys.append(pred_y)
    print('Oracle prediciton:', np.mean(pred_ys == true_y))
    return np.mean(pred_ys == true_y)

def _per_run(args, normal_dataset):
    print('current params:', args)
    train_stats, learner = train_main(args, normal_dataset, relative_error_ind=True)
    torch.save({'args':args, 'train_stats':train_stats, 'learner':learner}, f'./save_subgroups_new/subgroups_groups{args["set_groups"]}_ablatealpha{args["ablate_alpha"]}_exp1.pt')

if __name__ == '__main__':
    args = {
        'feature_dim': 10,
        'metric_rank': 10,
        'num_items': 100,
        'num_users': 50,
        'num_pairs_per_user': 500,
        'samples_per_user': 300,
        'noise_type': 'logistic',
        'noise_beta': 3,
        'num_groups': 8,
        'epochs': 200,
        'bs': 64,
        'lr': 0.005,
        'weight_decay_us': 0,
        'weight_decay_net': 0.001,
        'weight_decay_unconstrained_weight': 0,
        'optimizer_name': 'adam',
        'ablate_alpha': False,
        'ablate_m': False,
        'ablate_u': False,
    }
    normal_dataset = get_dataset(args)
    oracle_pred(normal_dataset)

    set_groups = list(range(2,101,2))
    ablate_alphas = [True,False]

    param_combinations = list(itertools.product(set_groups, ablate_alphas))

    args_list = []
    for i in param_combinations:
        args_tmp = deepcopy(args)
        args_tmp['set_groups']=i[0]
        args_tmp['ablate_alpha']=i[1]
        args_list.append(args_tmp)
    
    Parallel(n_jobs=32)(delayed(_per_run)(i,normal_dataset) for i in args_list)