import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

import dataset
import predict
import tools
from colors import *


def prediction_experiments_plot(filename, cfg):
    
    exp_matrix = predict.get_experiment_matrix(cfg)
    print(exp_matrix)
    train_sizes = []
    for i in exp_matrix:  # get training sizes
        split_id = exp_matrix[i][0]
        train_fps, _ = dataset.get_train_filepaths(cfg, split_id)
        train_sizes.append(len(train_fps))

    prediction_results = tools.load_json(os.path.join('results', filename))

    ind = np.arange(len(prediction_results['taa'])) 
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(13,6))
    fig.subplots_adjust(bottom=0.2, top=0.85)
    twin1 = ax.twinx()
    
    ta = ax.bar(ind, prediction_results['taa'], width, yerr=prediction_results['tas'], color=blue, error_kw={'elinewidth': 1})
    wf1 = ax.bar(ind+width, prediction_results['wfa'], width, yerr=prediction_results['wfs'], color=green, error_kw={'elinewidth': 1})
    ts = twin1.bar(ind+width*2, train_sizes, width, color=orange, error_kw={'elinewidth': 1})

    ax.set_ylim(0.5, 1)
    ax.set_xlabel('Experiment #')
    ax.set_ylabel('Performance metrics')
    twin1.set_ylabel('Train size')    
    ax.set_xticks(ind+width, exp_matrix)
    ax.legend((ta, wf1, ts), ('test acc', 'weight f1', 'train size'),
               ncol=3, bbox_to_anchor=(0.5, 1.02), loc='lower center',
               handletextpad=0.1, frameon=False)
    ax.margins(x=0.01)
    plt.show()
    # plt.savefig(os.path.join('results', 'prediction_summary'))
    # plt.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    prediction_experiments_plot('prediction_results.json', cfg)
