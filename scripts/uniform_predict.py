import argparse

import torch
import yaml

import predict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config.yaml')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.config, 'r'))

exp_matrix = predict.get_experiment_matrix(cfg)  # get the full matrix
exp_matrix_A = {k: exp_matrix.get(k, None) for k in (0, 1, 2)}  # subset trained on split A (RR only)
predict.prediction_experiments(cfg, device, exp_matrix_A, 'prediction_results_uniform.json')
