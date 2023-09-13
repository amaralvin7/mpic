import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def time_sync():
    '''
        Pytorch-accurate time. See: https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
    '''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def write_json(data, filename):

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def load_json(filename):

    with open(filename, 'r') as file:
        data = json.load(file)

    return data


def load_metadata():

    df = pd.read_csv('../../../../../mnt/ssd-cluster/vinicius/metadata.csv')
    
    return df
