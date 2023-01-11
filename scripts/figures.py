import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from itertools import product
from scipy.spatial.distance import pdist, squareform

import src.dataset as dataset
import src.predict as predict
import src.tools as tools


def prediction_summary(cfg, prediction_results_fname):

    exp_matrix = predict.get_experiment_matrix(cfg)
    train_sizes = []

    for i in exp_matrix:  # get training sizes
        split_id = exp_matrix[i][0]
        train_fps, _ = dataset.get_train_filepaths(cfg, split_id)
        train_sizes.append(len(train_fps))

    prediction_results = tools.load_json(os.path.join(
        '..', 'results', prediction_results_fname))

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
    plt.savefig(os.path.join('..', 'results', 'prediction_summary.pdf'))
    plt.close()


def get_exp_ids(cfg):
    
    exp_matrix = predict.get_experiment_matrix(cfg)

    ax_exp = {0: (('RR', 'RR_SRT', 'RR_FK', 'RR_SRT_FK'), 'RR'),
                  1: (('SRT', 'RR_SRT', 'SRT_FK', 'RR_SRT_FK'), 'SRT'),
                  2: (('FK', 'RR_FK', 'SRT_FK', 'RR_SRT_FK'), 'FK'),
                  3: (('SRT', 'FK', 'SRT_FK'), 'RR'),
                  4: (('RR', 'FK', 'RR_FK'), 'SRT'),
                  5: (('RR', 'SRT', 'RR_SRT'), 'FK')}
    
    exp_id_dict = {}

    for i in ax_exp:
        train_test_list = [[j, ax_exp[i][1]] for j in ax_exp[i][0]]
        l = []
        for j, k in product(exp_matrix, train_test_list):
            if exp_matrix[j] == k:
                l.append(j)
        exp_id_dict[i] = l            
            
    return exp_id_dict


def prediction_subplots_bar(cfg, prediction_results_fname):
    
    def get_ticklabels(exp_ids):
        
        labels = []
        for i in exp_ids:
            train_domains = exp_matrix[i][0].split('_')
            labels.append('\n'.join(train_domains))

        return labels
    
    exp_matrix = predict.get_experiment_matrix(cfg)
    train_sizes = []

    for i in exp_matrix:  # get training sizes
        split_id = exp_matrix[i][0]
        domains = split_id.split('_')
        train_size = 0
        for d in domains:
            train_fps, _ = dataset.get_train_filepaths(cfg, [d])
            train_size += len(train_fps)
        train_sizes.append(train_size)

    prediction_results = tools.load_json(os.path.join(
        '..', 'results', prediction_results_fname))

    ind = np.arange(len(prediction_results['taa']))
    width = 0.25

    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.1)
    
    exp_id_dict = get_exp_ids(cfg)

    for i, ax in enumerate(axs.flatten()):
        twin = ax.twinx()
        exp_ids = exp_id_dict[i]
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
    plt.savefig(os.path.join('..', 'results/prediction_subplots.pdf'))
    plt.close()
    

def prediction_subplots_scatter(cfg, prediction_results_fname):
    
    def get_df_domains(exp_ids):
        
        test_domain = exp_matrix[exp_ids[0]][1]
        domains_by_exp = []

        for i in exp_ids:
            train_domains = exp_matrix[i][0].split('_')
            domains_by_exp.append('_'.join(train_domains))

        return domains_by_exp, test_domain
    
    exp_matrix = predict.get_experiment_matrix(cfg)
    train_sizes = []

    for i in exp_matrix:  # get training sizes
        split_id = exp_matrix[i][0]
        domains = split_id.split('_')
        train_size = 0
        for d in domains:
            train_fps, _ = dataset.get_train_filepaths(cfg, [d])
            train_size += len(train_fps)
        train_sizes.append(train_size)

    prediction_results = tools.load_json(os.path.join(
        '..', 'results', prediction_results_fname))

    exp_id_dict = get_exp_ids(cfg)

    # train size
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
    
    for i, ax in enumerate(axs.flatten()):
        exp_ids = exp_id_dict[i]     
        taa = [prediction_results['taa'][j] for j in exp_ids]
        wfa = [prediction_results['wfa'][j] for j in exp_ids]
        ts = [train_sizes[j] for j in exp_ids]
        ta_plt = ax.scatter(ts, taa, c=blue, marker='o')
        wf_plt = ax.scatter(ts, wfa, c=green, marker='^')
        ax.set_xlim((0, 12000))
        if i < 3:
            ax.set_ylim((0.88, 1))
            ax.set_xticklabels([])
        else:
            ax.set_ylim((0.5, 1))
        if i not in (0, 3):
            ax.set_yticklabels([])          
                    
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SRT', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    fig.legend((ta_plt, wf_plt), ('test acc', 'weight f1'), loc='lower center', ncol=2, frameon=False)
    plt.savefig(os.path.join('..', 'results', 'prediction_subplots_scatter_trainsize.pdf'))
    plt.close()

    # distance
    df = distribution_heatmap(cfg, 'braycurtis', False)
    fig, axs = plt.subplots(2, 3, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axs.flatten()):

        exp_ids = exp_id_dict[i]
        train_domains, test_domain = get_df_domains(exp_ids)
        distances = df.loc[test_domain][train_domains]
        taa = [prediction_results['taa'][j] for j in exp_ids]
        wfa = [prediction_results['wfa'][j] for j in exp_ids]
        ta_plt = ax.scatter(distances, taa, c=blue, marker='o')
        wf_plt = ax.scatter(distances, wfa, c=green, marker='^')
        ax.set_xlim((-0.05, 0.8))
        if i < 3:
            ax.set_ylim((0.88, 1))
            ax.set_xticklabels([])
        else:
            ax.set_ylim((0.5, 1))
        if i not in (0, 3):
            ax.set_yticklabels([])         
    
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SRT', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    fig.legend((ta_plt, wf_plt), ('test acc', 'weight f1'), loc='lower center', ncol=2, frameon=False)
    plt.savefig(os.path.join('..', 'results', 'prediction_subplots_scatter_BCdistance.pdf'))
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
        plt.savefig(os.path.join('..', 'results', f'accuracy_{replicate_id}.png'))
        plt.close()

        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.plot(range(1, num_epochs + 1), model_output['train_loss_hist'],
                 label='training')
        plt.plot(range(1, num_epochs + 1), model_output['val_loss_hist'],
                 label='validation')
        plt.legend()
        plt.savefig(os.path.join('..', 'results', f'loss_{replicate_id}.png'))
        plt.close()


def get_class_count_df(cfg, normalize=False):

    classes = cfg['classes']
    split_ids = dataset.powerset(cfg['train_domains'])
    domains = cfg['train_domains']
    domain_splits = tools.load_json(os.path.join('..', 'data', cfg['domain_splits_fname']))
    
    df_list = []
    for d in domains:  # first, get all of the counts in the individual domains
        counts_by_label = []
        for c in classes:
            counts_by_label.append(sum(c in s for s in domain_splits[d]['train']))
        df_list.append(pd.DataFrame(counts_by_label, index=classes, columns=[d]))
    df = pd.concat(df_list, axis=1)

    for i in split_ids:
        if len(i) > 1:
            df[('_').join(i)] = df[list(i)].sum(axis=1)
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
        plt.savefig(os.path.join('..', 'results', f'distribution_heatmap_{metric}.pdf'))
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
    plt.savefig(os.path.join('..', 'results', f'distribution_barplot_{suffix}.pdf'))
    plt.close()


def uniform_comparison_barplots(cfg, ablation_predictions, uniform_predictions):

    exp_matrix = predict.get_experiment_matrix(cfg)
    test_domains = [exp_matrix[i][1] for i in (0, 1, 2)]

    train_fps, _ = dataset.get_train_filepaths(cfg, ('RR',))
    nonuniform_train_size = len(train_fps)
    
    # find the smallest number of observations that a class has
    train_fps_by_class = [[f for f in train_fps if c in f] for c in cfg['classes']]
    uniform_train_size = len(min(train_fps_by_class, key=len)) * len(cfg['classes'])

    a_predictions = tools.load_json(os.path.join(
        '..', 'results', ablation_predictions))
    u_predictions = tools.load_json(os.path.join(
        '..', 'results', uniform_predictions))
    
    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.5)
    width = 0.25
    
    ind = (0, 1)
    for i, (ax, test_domain) in enumerate(zip(axs, test_domains)):
        twin = ax.twinx()
        ax.set_xticks(ind, ('NU', 'U'))
        ax.grid(visible=True, which='major', axis='y', zorder=1)
        ax.bar(ind[0] - width, a_predictions['taa'][i],  width, yerr=a_predictions['tas'][i], color=blue, error_kw={'elinewidth': 1}, zorder=10)
        ax.bar(ind[0], a_predictions['wfa'][i],  width, yerr=a_predictions['wfs'][i], color=green, error_kw={'elinewidth': 1}, zorder=10)
        twin.bar(ind[0] + width, nonuniform_train_size,  width, color=orange, error_kw={'elinewidth': 1}, zorder=10)
        ta_bar_u = ax.bar(ind[1] - width, u_predictions['taa'][i],  width, yerr=u_predictions['tas'][i], color=blue, error_kw={'elinewidth': 1}, zorder=10)
        wf_bar_u = ax.bar(ind[1], u_predictions['wfa'][i],  width, yerr=u_predictions['wfs'][i], color=green, error_kw={'elinewidth': 1}, zorder=10)
        ts_bar_u = twin.bar(ind[1] + width, uniform_train_size,  width, color=orange, error_kw={'elinewidth': 1}, zorder=10)
        ax.set_ylim((0.4, 1))
        twin.set_ylim((1000, 7000))
        if i == 1:
            ax.set_yticklabels([])
            twin.set_yticklabels([])
        ax.set_title(test_domain, fontsize=14)
    fig.legend((ta_bar_u[0], wf_bar_u[0], ts_bar_u[0]), ('test acc', 'weight f1', 'train size'), loc='lower center', ncol=3, frameon=False)
    plt.savefig(os.path.join('..', 'results', 'uniform_comparison_barplots.pdf'))
    plt.close()

if __name__ == '__main__':

    black = '#000000'
    orange = '#E69F00'
    sky = '#56B4E9'
    green = '#009E73'
    blue = '#0072B2'
    vermillion = '#D55E00'
    radish = '#CC79A7'
    white = '#FFFFFF'

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(os.path.join('..', args.config), 'r'))
    
    ablation_predictions = 'prediction_results_ablations.json'
    uniform_predictions = 'prediction_results_uniform.json'

    # prediction_subplots_bar(cfg, ablation_predictions)
    # distribution_heatmap(cfg, 'cityblock', True)
    # distribution_heatmap(cfg, 'braycurtis', True)
    # distribution_barplot(cfg)
    # distribution_barplot(cfg, True)
    # prediction_subplots_scatter(cfg, ablation_predictions)
    uniform_comparison_barplots(cfg, ablation_predictions, uniform_predictions)
