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

exp_matrix = predict.get_experiment_matrix(cfg)  # get the full matrix
exp_matrix_RR = {k: exp_matrix.get(k, None) for k in (0, 1, 2)}  # subset trained on split RR only
predict.prediction_experiments(cfg, device, exp_matrix_RR, 'prediction_results_uniform.json', uniform=True)
