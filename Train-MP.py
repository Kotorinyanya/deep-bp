#!/usr/bin/env python
# coding: utf-8

# In[37]:


import os.path as osp
from itertools import repeat

import sklearn
from functools import partial
import random

import torch
import torch.distributed as dist
from torch_geometric.nn import DataParallel
from torch_geometric.data import Batch
from boxx import timeit
from sklearn.model_selection import KFold
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
from torch_geometric.datasets import TUDataset

from utils import *
import time
import numpy as np

from dataset import BAvsER, SBM4
from models.classification import Net
from models.community_detection import CD_BP_Net, CD_GCN_Net
from trainer import *
import argparse
import itertools

# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# In[38]:


import multiprocessing as mp

# Required positional argument
parser.add_argument('names', nargs='+', )
parser.add_argument('--device_ids', nargs='+', type=int, default=list(range(8)))

# Optional argument
parser.add_argument('--num_parallel', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=60)

# In[39]:


# names = [
#     'MUTAG', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI'
# ]

q_dict = {
    'MUTAG': 4,
    'ENZYMES': 15,
    'PROTEINS': 6,
    'IMDB-BINARY': 5,
    'IMDB-MULTI': 5,
}


# In[43]:


# In[44]:


def train_100(*args, **kwargs):
    print("training 100 in", args, kwargs)
    for _ in range(100):
        try:
            train_cross_validation(*args, **kwargs)
        except:
            pass


# In[45]:

# In[46]:

def get_jobs(names, device_ids, num_parallel=1, batch_size=1):
    folds = list(range(1, 11))
    jobs = [[[] for _ in range(len(folds))] for _ in range(len(device_ids))]
    for name in names[:1]:
        dataset = TUDataset(root='datasets/' + name, name=name)
        pad_dim = find_x_dim_max(dataset)
        trans = partial(pad_with_zero, pad_dim)
        dataset = TUDataset(root='datasets/' + name, name=name,
                            transform=trans)
        dataset.data.edge_attr = None
        model = Net
        iter_item = itertools.cycle(folds)
        for i, device_id in enumerate(device_ids[:]):
            job_count = 0
            for fold in iter_item:
                args = (model, dataset, 10)
                kwargs = dict(comment='debugbp3-' + name, batch_size=batch_size,
                              device_ids=[device_id], c_reg=0.01, fold_no=fold,
                              base_log_dir='runs_debug_ENZYMES')
                p = mp.Process(target=train_100,
                               args=args,
                               kwargs=kwargs)
                jobs[i][fold - 1].append(p)

                job_count += 1
                if job_count >= num_parallel:
                    break

    return jobs


# In[47]:

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    args = parser.parse_args()
    jobs = get_jobs(args.names, args.device_ids, args.num_parallel, args.batch_size)
    try:
        for device_jobs in jobs:
            for fold_jobs in device_jobs:
                for job in fold_jobs:
                    job.start()
                    time.sleep(2)

        for device_jobs in jobs:
            for fold_jobs in device_jobs:
                for job in fold_jobs:
                    job.join()
    except KeyboardInterrupt:
        for device_jobs in jobs:
            for fold_jobs in device_jobs:
                for job in fold_jobs:
                    if job.is_alive():
                        job.terminate()
