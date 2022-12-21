import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy.spatial.distance import pdist, squareform

import dataset
import predict
import tools
import torch
from colors import *


def prediction_summary(cfg):

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


def prediction_subplots_bar(cfg):
    
    def get_ticklabels(exp_ids):
        
        labels = []
        for i in exp_ids:
            train_domains = cfg['train_splits'][exp_matrix[i][0]]
            labels.append('\n'.join(train_domains))

        return labels
    
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

    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.1)

    ax_exp = {0: (0, 3, 6, 9), 3: (12, 15, 18), 1: (13, 4, 19, 10),
              4: (1, 16, 7), 2: (17, 8, 20, 11), 5: (2, 14, 5)} 
    
    for i, ax in enumerate(axs.flatten()):
        twin = ax.twinx()
        exp_ids = ax_exp[i]
        ind = np.arange(len(exp_ids))
        ax.set_xticks(ind + width, get_ticklabels(exp_ids))
        ax.grid(visible=True, which='major', axis='y', zorder=1)
        taa = [prediction_results['taa'][j] for j in exp_ids]
        tas = [prediction_results['tas'][j] for j in exp_ids]
        wfa = [prediction_results['wfa'][j] for j in exp_ids]
        wfs = [prediction_results['wfs'][j] for j in exp_ids]
        ts = [train_sizes[j] for j in exp_ids]
        ta_bar = ax.bar(ind, taa,  width, yerr=tas, color=blue, error_kw={'elinewidth': 1}, zorder=10)
        wf_bar = ax.bar(ind + width, wfa,  width, yerr=wfs, color=green, error_kw={'elinewidth': 1}, zorder=10)
        ts_bar = twin.bar(ind + width * 2, ts,  width, color=orange, error_kw={'elinewidth': 1}, zorder=10)
        ax.set_ylim((0.4, 1))
        twin.set_ylim((0, 12000))
        if i == 2 or i == 5:
            ax.set_yticklabels([])
        elif i == 0 or i == 3:
            twin.set_yticklabels([])
        else:
            ax.set_yticklabels([])
            twin.set_yticklabels([])
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SRT', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    fig.legend((ta_bar[0], wf_bar[0], ts_bar[0]), ('test acc', 'weight f1', 'train size'), loc='lower center', ncol=3, frameon=False)
    plt.savefig(os.path.join('results/prediction_subplots.pdf'))
    plt.close()
    

def prediction_subplots_scatter(cfg):
    
    def get_df_domains(exp_ids):
        
        test_domain = exp_matrix[0][1]
        domains_by_exp = []

        for i in exp_ids:
            train_domains = cfg['train_splits'][exp_matrix[i][0]]
            domains_by_exp.append('_'.join(train_domains))

        return domains_by_exp, test_domain
    
    exp_matrix = predict.get_experiment_matrix(cfg)
    train_sizes = []

    for i in exp_matrix:  # get training sizes
        split_id = exp_matrix[i][0]
        train_fps, _ = dataset.get_train_filepaths(cfg, split_id)
        train_sizes.append(len(train_fps))

    prediction_results = tools.load_json(os.path.join(
        'results', cfg['prediction_results_fname']))

    ax_exp = {0: (0, 3, 6, 9), 3: (12, 15, 18), 1: (13, 4, 19, 10),
              4: (1, 16, 7), 2: (17, 8, 20, 11), 5: (2, 14, 5)} 

    # train size
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.1)
    
    for i, ax in enumerate(axs.flatten()):
        exp_ids = ax_exp[i]       
        taa = [prediction_results['taa'][j] for j in exp_ids]
        wfa = [prediction_results['wfa'][j] for j in exp_ids]
        ts = [train_sizes[j] for j in exp_ids]
        ta_plt = ax.scatter(ts, taa, c=blue, marker='o')
        wf_plt = ax.scatter(ts, wfa, c=green, marker='^')        
        if i not in (0, 3):
            ax.set_yticklabels([])
    
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SRT', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    fig.legend((ta_plt, wf_plt), ('test acc', 'weight f1'), loc='lower center', ncol=2, frameon=False)
    plt.savefig(os.path.join('results/prediction_subplots_scatter_train.pdf'))
    plt.close()

    # cityblock
    for metric in ('cityblock', 'braycurtis'):
        df = distribution_heatmap(cfg, metric, False)

        fig, axs = plt.subplots(2, 3, figsize=(8, 6))
        fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.5)

        for i, ax in enumerate(axs.flatten()):
            exp_ids = ax_exp[i]
            train_domains, test_domain = get_df_domains(exp_ids)
            distances = df.loc[test_domain][train_domains]
            taa = [prediction_results['taa'][j] for j in exp_ids]
            wfa = [prediction_results['wfa'][j] for j in exp_ids]
            ta_plt = ax.scatter(distances, taa, c=blue, marker='o')
            wf_plt = ax.scatter(distances, wfa, c=green, marker='^')        
        
        axs.flatten()[0].set_title('RR', fontsize=14)
        axs.flatten()[1].set_title('SRT', fontsize=14)
        axs.flatten()[2].set_title('FK', fontsize=14)
        fig.legend((ta_plt, wf_plt), ('test acc', 'weight f1'), loc='lower center', ncol=2, frameon=False)
        plt.savefig(os.path.join(f'results/prediction_subplots_scatter_{metric}.pdf'))
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


def get_class_count_df(cfg, normalize=False):

    classes = cfg['classes']
    split_ids = cfg['train_splits']
    domains = cfg['train_domains']
    domain_splits = tools.load_json(cfg['domain_splits_fname'])
    
    df_list = []
    for d in domains:  # first, get all of the counts in the individual domains
        counts_by_label = []
        for c in classes:
            counts_by_label.append(sum(c in s for s in domain_splits[d]['train']))
        df_list.append(pd.DataFrame(counts_by_label, index=classes, columns=[d]))
    df = pd.concat(df_list, axis=1)

    for i in split_ids:
        if len(split_ids[i]) > 1:
            df[('_').join(split_ids[i])] = df[split_ids[i]].sum(axis=1)
    if normalize:
        for c in df.columns:
            df[c] = df[c] / df[c].sum()
    
    return df

def distribution_heatmap(cfg, metric='cityblock', make_fig=False):
    
    df = get_class_count_df(cfg, normalize=True)
    df = pd.DataFrame(squareform(pdist(df.T, metric=metric)), columns=df.columns, index=df.columns)
    if make_fig:
        ax = sns.heatmap(df, cmap='viridis', annot=True, fmt='.2f')
        ax.figure.tight_layout()
        plt.savefig('results/distribution_heatmap.pdf')
        plt.close()
    else:
        return df

def distribution_barplot(cfg, normalize=False):
    
    df = get_class_count_df(cfg, normalize=normalize)
    ind = np.arange(len(df)) 
    width = 0.25
    bar1 = plt.bar(ind, df['RR'], width, color=blue)
    bar2 = plt.bar(ind+width, df['SRT'], width, color=green)
    bar3 = plt.bar(ind+width*2, df['FK'], width, color=orange)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(ind+width, df.index.values)
    plt.xticks(rotation=45, ha='right')
    plt.legend((bar1, bar2, bar3), ('RR', 'SRT', 'FK'), ncol=3, bbox_to_anchor=(0.5, 1.02), loc='lower center',
                handletextpad=0.1, frameon=False)

    if normalize:
        ylabel = 'Fraction of observations'
        suffix = 'fractions'
    else:
        ylabel = 'Number of observations'
        suffix = 'totals'
        plt.yscale('log')
    plt.ylabel(ylabel)
    plt.savefig(f'results/distribution_barplot_{suffix}.pdf')
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))

    # prediction_subplots_bar(cfg)
    # distribution_heatmap(cfg)
    # distribution_barplot(cfg)
    # distribution_barplot(cfg, True)
    prediction_subplots_scatter(cfg)
