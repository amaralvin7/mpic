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

train_fps, val_fps = dataset.compile_domain_filepaths(cfg, ('RR',), cfg['ablation_classes'])

# separate_train_fps by class
train_fps_by_class = [[f for f in train_fps if c in f] for c in cfg['ablation_classes']]

# find the smallest number of observations that a class has
n = len(min(train_fps_by_class, key=len))

# subsample n instances from each class and concatenate them
train_fps_uniform = []
for l in train_fps_by_class:
    train_fps_uniform.extend(l[:n])
mean, std = dataset.calculate_data_stats(cfg, train_fps_uniform, cfg['ablation_classes'])

# train replicates
model_name = 'RRu'
print(f'---------Training Model {model_name}...')
for i in range(cfg['replicates']):
    output = train.train_model(cfg, device, train_fps_uniform, val_fps, mean, std, i)
    replicate_id = f'{model_name}-{i}'
    torch.save(
        output,
        os.path.join('..', 'weights', f'saved_model_{replicate_id}.pt'))