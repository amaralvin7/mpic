import argparse
import os

import yaml

import src.tools as tools
import src.train as train

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_list_id', '-i')
args = parser.parse_args()
list_id = int(args.model_name_list_id)

model_names = tools.get_model_names(list_id)

for model_name in model_names:
    cfg = yaml.safe_load(open(f'../configs/{model_name}.yaml', 'r'))
    replicates = 5
    for replicate in range(replicates):
        tools.set_seed(replicate)
        model_replicate = f'{model_name}-{replicate}'
        train.train_model(cfg, model_replicate, list_id)

