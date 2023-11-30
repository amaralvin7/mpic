import json
import os
import random
import shutil
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
    
    id_dict = {0: ('targetRR_ood', 'highLR', 'highWD', 'lowLR', 'lowWD', 'normdata',
                   'normIN', 'pad', 'padnormdata', 'padnormIN'),
               1: ('targetRR_ood',),
               2: ('targetRR_top1k', 'targetRR_verify', 'targetRR_minboost'),
               3: ('targetJC_ood',),
               4: ('targetJC_top1k', 'targetJC_verify', 'targetJC_minboost')}

    return id_dict[list_id]


def copy_class_imgs(c, filepath_list, copy_from, copy_to):

    os.makedirs(f'{copy_to}/{c}')
    for i in filepath_list:
        old_filepath = f'{copy_from}/{i}'
        new_filepath = f'{copy_to}/{c}/{i.split("/")[1]}'
        shutil.copyfile(old_filepath, new_filepath)

