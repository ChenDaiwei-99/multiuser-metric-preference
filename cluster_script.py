DESCRIPTION = """
Clustering scripting for multi-user preference learning
"""
# thread settings
import os
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'

import multiprocessing as mp
from run_experiments import run_experiment
import time
import numpy as np
import yaml
import argparse

parser = argparse.ArgumentParser(DESCRIPTION)
parser.add_argument('--config', type=str, help='Configuration file path (.yaml)')
args = parser.parse_args()

# load config
with open(args.config) as f:
    config = yaml.safe_load(f)

save_root = config['save_root']
exp_name = config['exp_name']
n_cores = config['n_cores']
pm = config['pm']
methods = config['methods']

# setup seeding
now_ms = round(time.time() * 100)
np.random.seed(now_ms % 2**32)

# set parameters
params = [(pm, methods, '{}_{}_{}'.format(save_root + exp_name, now_ms, i),
           np.random.randint(2**31)) for i in range(n_cores)]

# run on cores
pool = mp.Pool(n_cores)
pool.starmap(run_experiment, params)