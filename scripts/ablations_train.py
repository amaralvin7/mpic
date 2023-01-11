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

train_splits = dataset.powerset(cfg['train_domains'])

for domains in train_splits:

    train_split_id = ('_').join(domains)
    train_fps, val_fps = dataset.get_train_filepaths(cfg, domains)
    mean, std = dataset.get_train_data_stats(cfg, train_split_id)

    print(f'---------Training Model {train_split_id}...')
    for i in range(cfg['replicates']):
        output = train.train_model(cfg, device, train_fps, val_fps, mean, std, i)
        replicate_id = f'{train_split_id}-{i}'
        torch.save(
            output,
            os.path.join('..', 'weights', f'saved_model_{replicate_id}.pt'))
        
        