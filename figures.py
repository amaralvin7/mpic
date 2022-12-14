import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

import dataset
import predict
import tools
import torch
from colors import *


def prediction_experiments_plot(cfg):

    exp_matrix = predict.get_experiment_matrix(cfg)
    train_sizes = []

    for i in exp_matrix:  # get training sizes
        split_id = exp_matrix[i][0]
        train_fps, _ = dataset.get_train_filepaths(cfg, split_id)
        train_sizes.append(len(train_fps))

    prediction_results = tools.load_json(os.path.join(
        'results', cfg['prediction_results_fname']))

    ind = np.arange(len(prediction_results['taa']))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.subplots_adjust(bottom=0.2, top=0.85)
    twin1 = ax.twinx()

    ta = ax.bar(ind, prediction_results['taa'],  width,
                yerr=prediction_results['tas'], color=blue,
                error_kw={'elinewidth': 1})
    wf1 = ax.bar(ind + width, prediction_results['wfa'], width,
        yerr=prediction_results['wfs'], color=green,
        error_kw={'elinewidth': 1})
    ts = twin1.bar(ind + width * 2, train_sizes, width, color=orange,
                   error_kw={'elinewidth': 1})

    ax.set_ylim(0.5, 1)
    ax.set_xlabel('Experiment #')
    ax.set_ylabel('Performance metrics')
    twin1.set_ylabel('Train size')
    ax.set_xticks(ind + width, exp_matrix)
    ax.legend((ta, wf1, ts), ('test acc', 'weight f1', 'train size'), ncol=3,
              bbox_to_anchor=(0.5,1.02), loc='lower center',
              handletextpad=0.1, frameon=False)
    ax.margins(x=0.01)
    plt.savefig(os.path.join('results', 'prediction_summary.pdf'))
    plt.close()


def training_plots():

    models = [f for f in os.listdir('weights') if f'.pt' in f]

    for m in models:
        split = m.split('_')
        replicate_id = '_'.join(split[2:]).split('.')[0]
        model_output = torch.load(os.path.join('weights', m),
                                  map_location='cpu')
        num_epochs = len(model_output['train_loss_hist'])

        plt.xlabel('Training Epochs')
        plt.ylabel('Accuracy')
        plt.plot(range(1, num_epochs + 1), model_output['train_acc_hist'],
                 label='training')
        plt.plot(range(1, num_epochs + 1), model_output['val_acc_hist'],
                 label='validation')
        plt.legend()
        plt.savefig(os.path.join('results', f'accuracy_{replicate_id}.png'))
        plt.close()

        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.plot(range(1, num_epochs + 1), model_output['train_loss_hist'],
                 label='training')
        plt.plot(range(1, num_epochs + 1), model_output['val_loss_hist'],
                 label='validation')
        plt.legend()
        plt.savefig(os.path.join('results', f'loss_{replicate_id}.png'))
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))

    prediction_experiments_plot(cfg)
    training_plots()
