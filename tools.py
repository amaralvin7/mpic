import random
import time

import numpy as np
import torch

def set_seed(cfg, device):
      
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if device != 'cpu':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

def time_sync():
    '''
        Pytorch-accurate time. See: https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
    '''
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()