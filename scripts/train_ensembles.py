import argparse
import os

import yaml

import src.tools as tools
import src.train as train

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', '-e')
args = parser.parse_args()
exp_name = args.experiment_name

cfgs = sorted([c for c in os.listdir(f'../configs/{exp_name}') if '.yaml' in c])

for cfg_fn in cfgs:
    cfg = yaml.safe_load(open(f'../configs/{exp_name}/{cfg_fn}', 'r'))
    if '-' in cfg_fn:  # if there are replicate cfg files
        replicate = int(cfg_fn.split('-')[1][0])
        tools.set_seed(0)
        model_replicate = cfg_fn.split('.')[0]
        train.train_model(cfg, model_replicate, exp_name)
    else:
        replicates = 5
        for replicate in range(replicates):
            tools.set_seed(replicate)
            model_replicate = f'{cfg_fn.split(".")[0]}-{replicate}'
            train.train_model(cfg, model_replicate, exp_name)

