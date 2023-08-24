import os

# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model
import torch
import yaml

import src.dataset as dataset
import src.tools as tools
import src.train as train
from src.priv import comet_key

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = yaml.safe_load(open(os.path.join('..', 'config.yaml'), 'r'))
tools.set_seed(cfg, device)
# experiment = Experiment(api_key=comet_key)
train_fps, val_fps = dataset.compile_trainval_filepaths(cfg, ('SR', 'JC', 'FC', 'FO'))
mean = None
std = None
pads = (True, False)

for p in pads:
    print(f'---------Training Models (pad={p})...')
    for i in range(cfg['replicates']):
        output = train.train_model(cfg, device, train_fps, val_fps, mean, std, i, p)
        replicate_id = f'pad{p}-{i}'
        torch.save(output, os.path.join('..', 'runs', 'pad_exp_targetRR', 'weights', f'savedmodel_{replicate_id}.pt'))
