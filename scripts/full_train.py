import argparse
import os

import torch
import yaml

import src.dataset as dataset
import src.tools as tools
import src.train as train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config.yaml')
args = parser.parse_args()
cfg = yaml.safe_load(open(os.path.join('..', args.config), 'r'))
tools.set_seed(cfg, device)

domains = cfg['domains']
train_fps, val_fps = dataset.compile_domain_filepaths(cfg, domains, cfg['all_classes'])
mean, std = dataset.get_data_stats()

print(f'---------Training Model...')
for i in range(cfg['replicates']):
    output = train.train_model(cfg, device, train_fps, val_fps, mean, std, i)
    replicate_id = f'FULL-{i}'
    torch.save(
        output,
        os.path.join('..', 'weights', f'saved_model_{replicate_id}.pt'))
        
        