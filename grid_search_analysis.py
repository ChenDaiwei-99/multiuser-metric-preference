import numpy as np
import matplotlib.pyplot as plt
import torch

import os 
import sys 
import re
from tqdm import tqdm
from joblib import Parallel, delayed

if __name__ == '__main__':
    grid_search_fileNames = os.listdir('./save/')
    grid_search_filePaths = [os.path.join('./save/',i) for i in grid_search_fileNames]

    train_stats_all = []

    def tmp_fun(filePath):
        res = torch.load(filePath)
        return({**res['args'], 'test_accu_record': res['train_stats']['test_accu_record']})

    train_stats_all = Parallel(n_jobs=8)(delayed(tmp_fun)(i) for i in tqdm(grid_search_filePaths))

    torch.save(train_stats_all, 'train_stats_all.pt')

