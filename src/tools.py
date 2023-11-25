import json
import random
import time

import numpy as np
import pandas as pd
import torch


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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

    df = pd.read_csv('../../mpic_data/metadata.csv')
    
    return df


def get_model_names(list_id):
    
    id_dict = {1: ('targetRR_ood',),
               2: ('targetRR_top1k', 'targetRR_verify', 'targetRR_minboost')}

    return id_dict[list_id]