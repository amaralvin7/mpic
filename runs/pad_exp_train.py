import os

import torch
import yaml

import src.dataset as dataset
import src.tools as tools
import src.train as train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg_filename = 'config.yaml'
cfg = yaml.safe_load(open(cfg_filename, 'r'))
tools.set_seed(cfg, device)

train_fps, val_fps = dataset.compile_trainval_filepaths(cfg, ('SR', 'RR', 'FC', 'FO'))
mean = None
std = None
pads = (True, False)

for p in pads:
    print(f'---------Training Models (pad={p})...')
    for i in range(cfg['replicates']):
        output = train.train_model(cfg, device, train_fps, val_fps, mean, std, i, p)
        replicate_id = f'pad{p}-{i}'
        torch.save(output, f'./weights_new/savedmodel_{replicate_id}.pt')
