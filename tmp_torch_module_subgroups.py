from group_dataset import Dataset
from metric_check import train_main

from joblib import Parallel, delayed
import torch
import os
import sys
from tqdm import tqdm

def _per_run(num_groups):
    ########### hyperparameters for the dataset ################
    feature_dim = 10
    metric_rank = 10
    num_items = 100
    num_users = 100
    num_pairs_per_user = 1000
    noise_type = 'logistic'
    noise_beta = 1
    num_groups = num_groups

    normal_dataset = Dataset(dataset_type='Normal',
                            d=feature_dim,
                            r=metric_rank,
                            n=num_items,
                            N=num_users,
                            m=num_pairs_per_user,
                            noise_type=noise_type,
                            noise_beta=noise_beta,
                            num_groups=num_groups)
    
    ########### initialize the dataloders ################
    normal_data = normal_dataset.getAllData()
    items, observations, true_y, true_M, true_u = normal_data['X'], normal_data['S'], normal_data['Y'], normal_data['M'], normal_data['U']
    true_y_noiseless = normal_dataset.Y_noiseless

    args = {
            'feature_dim': feature_dim,
            'metric_rank': metric_rank,
            'num_items': num_items,
            'num_users': num_users,
            'num_pairs_per_user': 500,
            'noise_type': noise_type,
            'noise_beta': noise_beta,
            'num_groups': num_groups,
    }

    train_stats, learner = train_main(args, normal_dataset, relative_error_ind=False)
    torch.save({
        'args': args,
        'train_stats': train_stats,
        'learner': learner,}, f'./save_subgroups/subgroups_numgroups{num_groups}.pt')


if __name__ == '__main__':
    results = Parallel(n_jobs=8)(delayed(_per_run)(num_groups) for num_groups in tqdm(range(5,51,5)))
    