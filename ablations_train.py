import argparse
import os

import torch
import yaml

import dataset
import tools
import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config.yaml')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config, 'r'))
tools.set_seed(cfg, device)

for split_id in cfg['train_splits']:

    train_fps, val_fps = dataset.get_train_filepaths(cfg, split_id)
    mean, std = dataset.get_train_data_stats(cfg, split_id)

    print(f'---------Training Model {split_id}...')
    for i in range(cfg['replicates']):
        output = train.train_model(cfg, device, train_fps, val_fps, mean, std, i)
        replicate_id = f'{split_id}_{i}'
        torch.save(
            output,
            os.path.join('weights', f'saved_model_{replicate_id}.pt'))