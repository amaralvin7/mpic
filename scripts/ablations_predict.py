import argparse
import os

import torch
import yaml

import src.predict as predict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config.yaml')
args = parser.parse_args()
cfg = yaml.safe_load(open(os.path.join('..', args.config), 'r'))

exp_matrix = predict.get_experiment_matrix(cfg)
predict.prediction_experiments(cfg, device, exp_matrix, 'prediction_results_ablations.json')
