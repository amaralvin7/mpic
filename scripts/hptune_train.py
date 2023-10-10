import os

import yaml
from itertools import product

import src.tools as tools
import src.train as train

replicates = 5
cfgs = sorted([c for c in os.listdir('../configs') if '.yaml' in c])

for cfg_fn, replicate in product(cfgs, range(replicates)):
    cfg = yaml.safe_load(open(f'../configs/{cfg_fn}', 'r'))
    tools.set_seed(replicate)
    exp_id = f'{cfg_fn.split(".")[0]}-{replicate}'
    train.train_model(cfg, exp_id)

