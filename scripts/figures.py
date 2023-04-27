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
from matplotlib.lines import Line2D
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

    ax_exp = {0: (('RR', 'RR_SR', 'RR_FK', 'RR_JC', 'RR_SR_FK', 'RR_SR_JC', 'RR_FK_JC', 'RR_SR_FK_JC'), 'RR'),
              1: (('SR', 'RR_SR', 'SR_FK', 'SR_JC', 'RR_SR_FK', 'RR_SR_JC', 'SR_FK_JC', 'RR_SR_FK_JC'), 'SR'),
              2: (('FK', 'RR_FK', 'SR_FK', 'FK_JC', 'RR_SR_FK', 'RR_FK_JC', 'SR_FK_JC', 'RR_SR_FK_JC'), 'FK'),
              3: (('JC', 'RR_JC', 'SR_JC', 'FK_JC', 'RR_SR_JC', 'RR_FK_JC', 'SR_FK_JC', 'RR_SR_FK_JC'), 'JC'),
              4: (('SR', 'FK', 'JC', 'SR_FK', 'SR_JC', 'FK_JC'), 'RR'),
              5: (('RR', 'FK', 'JC', 'RR_FK', 'RR_JC', 'FK_JC'), 'SR'),
              6: (('RR', 'SR', 'JC', 'RR_SR', 'RR_JC', 'SR_JC'), 'FK'),
              7: (('RR', 'SR', 'FK', 'RR_SR', 'RR_FK', 'SR_FK'), 'JC')}
    
    exp_id_dict = {}

    for i in ax_exp:
        train_test_list = [[j, ax_exp[i][1]] for j in ax_exp[i][0]]
        l = []
        for j, k in product(exp_matrix, train_test_list):
            if exp_matrix[j] == k:
                l.append(j)
        exp_id_dict[i] = l            
            
    return exp_id_dict


def prediction_subplots_bar(cfg, classes, prediction_results_fname):
    
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
            train_fps, _ = dataset.compile_domain_filepaths(cfg, [d], classes)
            train_size += len(train_fps)
        train_sizes.append(train_size)

    prediction_results = tools.load_json(os.path.join(
        '..', 'results', prediction_results_fname))

    ind = np.arange(len(prediction_results['taa']))
    width = 0.25

    fig, axs = plt.subplots(2, 4, figsize=(13, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.5, wspace=0.1)
    
    exp_id_dict = get_exp_ids(cfg)

    for i, ax in enumerate(axs.flatten()):
        twin = ax.twinx()
        exp_ids = exp_id_dict[i]
        ind = np.arange(len(exp_ids))
        ax.set_xticks(ind + width, get_ticklabels(exp_ids))
        ax.grid(visible=True, which='major', axis='y', zorder=1)
        taa = [prediction_results['taa'][j] for j in exp_ids]
        tas = [prediction_results['tas'][j] for j in exp_ids]
        # wfa = [prediction_results['wfa'][j] for j in exp_ids]
        # wfs = [prediction_results['wfs'][j] for j in exp_ids]
        ts = [train_sizes[j] for j in exp_ids]
        ta_bar = ax.bar(ind, taa,  width, yerr=tas, color=blue, error_kw={'elinewidth': 1}, zorder=10)
        # wf_bar = ax.bar(ind + width, wfa,  width, yerr=wfs, color=green, error_kw={'elinewidth': 1}, zorder=10)
        ts_bar = twin.bar(ind + width, ts,  width, color=orange, error_kw={'elinewidth': 1}, zorder=10)
        ax.set_ylim((0.2, 1))
        twin.set_ylim((0, 20000))
        if i == 3 or i == 7:
            ax.set_yticklabels([])
        elif i == 0 or i == 4:
            twin.set_yticklabels([])
        else:
            ax.set_yticklabels([])
            twin.set_yticklabels([])
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SR', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    axs.flatten()[3].set_title('JC', fontsize=14)
    # fig.legend((ta_bar[0], wf_bar[0], ts_bar[0]), ('test acc', 'weight f1', 'train size'), loc='lower center', ncol=3, frameon=False)
    fig.legend((ta_bar[0], ts_bar[0]), ('test acc', 'train size'), loc='lower center', ncol=2, frameon=False)
    plt.savefig(os.path.join('..', 'results/prediction_subplots.pdf'), bbox_inches='tight')
    plt.close()
    

def prediction_subplots_scatter(cfg, classes, prediction_results_fname):
    
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
            train_fps, _ = dataset.compile_domain_filepaths(cfg, [d], classes)
            train_size += len(train_fps)
        train_sizes.append(train_size)

    prediction_results = tools.load_json(os.path.join(
        '..', 'results', prediction_results_fname))

    exp_id_dict = get_exp_ids(cfg)

    # train size
    fig, axs = plt.subplots(2, 4, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
    
    for i, ax in enumerate(axs.flatten()):
        exp_ids = exp_id_dict[i]     
        taa = [prediction_results['taa'][j] for j in exp_ids]
        tas = [prediction_results['tas'][j] for j in exp_ids]
        # wfa = [prediction_results['wfa'][j] for j in exp_ids]
        ts = [train_sizes[j] for j in exp_ids]
        ta_plt = ax.errorbar(ts, taa, tas, c=blue, fmt='o', ecolor=black)
        # ta_plt = ax.scatter(ts, taa, c=blue, marker='o')
        # wf_plt = ax.scatter(ts, wfa, c=green, marker='^')
        ax.set_xlim((0, 18000))
        ax.set_ylim((0.3, 1.05))
        if i < 4:
            ax.set_xticklabels([])
        if i not in (0, 4):
            ax.set_yticklabels([])

    axs.flatten()[0].set_ylabel('Accuracy (in-domain)')
    axs.flatten()[4].set_ylabel('Accuracy (out-of-domain)')
    fig.text(0.5, 0.04, 'Training size', ha='center')      
                    
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SR', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    axs.flatten()[3].set_title('JC', fontsize=14)
    # fig.legend((ta_plt, wf_plt), ('test acc', 'weight f1'), loc='lower center', ncol=2, frameon=False)
    plt.savefig(os.path.join('..', 'results', 'prediction_subplots_scatter_trainsize.pdf'), bbox_inches='tight')
    plt.close()

    # distance
    df = distribution_heatmap(cfg, classes, 'braycurtis', False)
    fig, axs = plt.subplots(2, 4, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axs.flatten()):
        exp_ids = exp_id_dict[i]
        train_domains, test_domain = get_df_domains(exp_ids)
        distances = df.loc[test_domain][train_domains]
        taa = [prediction_results['taa'][j] for j in exp_ids]
        tas = [prediction_results['tas'][j] for j in exp_ids]
        # wfa = [prediction_results['wfa'][j] for j in exp_ids]
        ts = [train_sizes[j] for j in exp_ids]
        ta_plt = ax.errorbar(distances, taa, tas, c=blue, fmt='o', ecolor=black)
        # ta_plt = ax.scatter(ts, taa, c=blue, marker='o')
        # wf_plt = ax.scatter(distances, wfa, c=green, marker='^')
        ax.set_xlim((-0.05, 0.8))
        ax.set_ylim((0.3, 1.05))
        if i < 4:
            ax.set_xticklabels([])
        if i not in (0, 4):
            ax.set_yticklabels([])        
    
    axs.flatten()[0].set_ylabel('Accuracy (in-domain)')
    axs.flatten()[4].set_ylabel('Accuracy (out-of-domain)')
    fig.text(0.5, 0.04, 'Bray-Curtis distance', ha='center')
    
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SR', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    axs.flatten()[3].set_title('JC', fontsize=14)
    # fig.legend((ta_plt, wf_plt), ('test acc', 'weight f1'), loc='lower center', ncol=2, frameon=False)
    plt.savefig(os.path.join('..', 'results', 'prediction_subplots_scatter_BCdistance.pdf'), bbox_inches='tight')
    plt.close()


def training_plots():

    models = [f for f in os.listdir('../weights') if f'.pt' in f]

    for m in models:
        split = m.split('_')
        replicate_id = '_'.join(split[2:]).split('.')[0]
        model_output = torch.load(os.path.join('../weights', m),
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


def get_class_count_df(cfg, classes, normalize=False):

    split_ids = dataset.powerset(cfg['domains'])
    domain_splits = tools.load_json(os.path.join('..', 'data', cfg['domain_splits_fname']))
    
    df_list = []
    for d in cfg['domains']:  # first, get all of the counts in the individual domains
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

def distribution_heatmap(cfg, classses, metric='braycurtis', make_fig=False):
    
    df = get_class_count_df(cfg, classses, normalize=True)
    df = pd.DataFrame(squareform(pdist(df.T, metric=metric)), columns=df.columns, index=df.columns)
    if make_fig:
        ax = sns.heatmap(df, cmap='viridis', annot=True, fmt='.2f')
        ax.figure.set_size_inches(8, 6)
        ax.figure.tight_layout()
        plt.savefig(os.path.join('..', 'results', f'distribution_heatmap_{metric}.pdf'))
        plt.close()
    else:
        return df

def distribution_barplot(cfg, classses, normalize=False):
    
    df = get_class_count_df(cfg, classses, normalize=normalize)
    ind = np.arange(len(df)) 
    width = 0.2
    bar1 = plt.bar(ind-width*0.5, df['RR'], width, color=blue)
    bar2 = plt.bar(ind+width*0.5, df['SR'], width, color=green)
    bar3 = plt.bar(ind+width*1.5, df['FK'], width, color=orange)
    bar4 = plt.bar(ind+width*2.5, df['JC'], width, color=vermillion)
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(ind+width, df.index.values)
    plt.xticks(rotation=45, ha='right')
    plt.legend((bar1, bar2, bar3, bar4), ('RR', 'SR', 'FK', 'JC'), ncol=4, bbox_to_anchor=(0.5, 1.02), loc='lower center',
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
    test_domains = [exp_matrix[i][1] for i in (0, 1, 2, 3)]

    train_fps, _ = dataset.compile_domain_filepaths(cfg, ('RR',), cfg['ablation_classes'])
    nonuniform_train_size = len(train_fps)
    
    # find the smallest number of observations that a class has
    train_fps_by_class = [[f for f in train_fps if c in f] for c in cfg['ablation_classes']]
    uniform_train_size = len(min(train_fps_by_class, key=len)) * len(cfg['ablation_classes'])

    a_predictions = tools.load_json(os.path.join(
        '..', 'results', ablation_predictions))
    u_predictions = tools.load_json(os.path.join(
        '..', 'results', uniform_predictions))
    
    fig, axs = plt.subplots(1, 4, figsize=(8, 4))
    fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.5)
    width = 0.25
    
    ind = (0, 1)
    for i, (ax, test_domain) in enumerate(zip(axs, test_domains)):
        twin = ax.twinx()
        ax.set_xticks(ind, ('NU', 'U'))
        ax.grid(visible=True, which='major', axis='y', zorder=1)
        ax.bar(ind[0] - 0.5*width, a_predictions['taa'][i],  width, yerr=a_predictions['tas'][i], color=blue, error_kw={'elinewidth': 1}, zorder=10)
        # ax.bar(ind[0], a_predictions['wfa'][i],  width, yerr=a_predictions['wfs'][i], color=green, error_kw={'elinewidth': 1}, zorder=10)
        twin.bar(ind[0] + 0.5*width, nonuniform_train_size,  width, color=orange, error_kw={'elinewidth': 1}, zorder=10)
        ta_bar_u = ax.bar(ind[1] - 0.5*width, u_predictions['taa'][i],  width, yerr=u_predictions['tas'][i], color=blue, error_kw={'elinewidth': 1}, zorder=10)
        # wf_bar_u = ax.bar(ind[1], u_predictions['wfa'][i],  width, yerr=u_predictions['wfs'][i], color=green, error_kw={'elinewidth': 1}, zorder=10)
        ts_bar_u = twin.bar(ind[1] + 0.5*width, uniform_train_size,  width, color=orange, error_kw={'elinewidth': 1}, zorder=10)
        ax.set_ylim((0.4, 1))
        twin.set_ylim((1000, 7000))
        if i == 0:
            twin.set_yticklabels([])
        elif i < len(axs) - 1:
            ax.set_yticklabels([])
            twin.set_yticklabels([])
        else:
            ax.set_yticklabels([])
        ax.set_title(test_domain, fontsize=14)
    # fig.legend((ta_bar_u[0], wf_bar_u[0], ts_bar_u[0]), ('test acc', 'weight f1', 'train size'), loc='lower center', ncol=3, frameon=False)
    fig.legend((ta_bar_u[0], ts_bar_u[0]), ('test acc', 'train size'), loc='lower center', ncol=3, frameon=False)
    plt.savefig(os.path.join('..', 'results', 'uniform_comparison_barplots.pdf'))
    plt.close()


def get_domain_color(domain):
    
    if domain == 'SR':
        c = green
    elif domain == 'FK':
        c = orange
    elif domain  == 'JC':
        c = vermillion
    else:
        c = blue
    
    return c


def particle_volume(p_class, esd, w_params=None):
    
    if p_class == 'sphere':
        w = esd
        l = esd
        v = (4/3) * np.pi * (esd/2)**3        
    elif p_class == 'cylinder':
        w = (w_params[0] * esd) / (esd + w_params[1])
        l = np.pi * (esd/2)**2 / w
        v = l * np.pi * (w/2)**2
    elif p_class == 'ellipsoid':
        w = 0.54 * esd
        l = esd**2 / w
        v = (4/3) * (l/2) * np.pi * (w/2)**2
    elif p_class == 'cuboid':
        w = 0.63 * esd
        l = np.pi * (esd/2)**2 / w
        v = l * w * (w/4)
    else:
        print('UNIDENTIFIED PARTICLE in particle_volume')
        
    return v


def shape_to_flux(a, b, v, area, time):
    
    carbon =  (a * v**b) / 12.011  # mg to mmol
    flux = carbon / (area * time)
    
    return flux


def flux_equations(row, col):

    p_class = row[col]
    esd = row['ESD']  # µm
    area = row['area']  # m-2
    time = row['time']  # d-1

    if p_class in ('aggregate', 'mini_pellet', 'rhizaria', 'phytoplankton'):
        v = particle_volume('sphere', esd)
        if p_class == 'aggregate':
            a = 0.07 * 10**-9
            b = 0.83
        elif p_class == 'mini_pellet':
            a = 0.07 * 10**-9
            b = 1
        elif p_class == 'rhizaria':
            a = 0.004 * 10**-9
            b = 0.939
        else:  # phytoplankton
            a = 0.288 * 10**-9
            b = 0.811
    elif p_class == 'long_pellet':
        v = particle_volume('cylinder', esd, (264, 584))
        a = 0.07 * 10**-9
        b = 1
    elif p_class == 'short_pellet':
        v = particle_volume('ellipsoid', esd)
        a = 0.07 * 10**-9
        b = 1
    elif p_class == 'salp_pellet':
        v = particle_volume('cuboid', esd)
        a = 0.04 * 10**-9
        b = 1
    elif p_class in ('fiber', 'swimmer', 'unidentifiable'):
        return 0
    else:
        print('UNKNOWN PARTICLE TYPE')
        sys.exit()
    
    return shape_to_flux(a, b, v, area, time)


def commonize_manual_labels():
    
    d = {'aggregate': 'aggregate', 'amphipod': 'swimmer', 'copepod': 'swimmer',
         'dense_detritus': 'aggregate', 'fiber': 'fiber',
         'foraminifera': 'swimmer', 'large_loose_pellet': 'long_pellet',
         'long_fecal_pellet': 'long_pellet', 'mini_pellet': 'mini_pellet',
         'phytoplankton': 'phytoplankton', 'pteropod': 'swimmer',
         'rhizaria': 'rhizaria', 'salp_pellet': 'salp_pellet',
         'short_pellet': 'short_pellet', 'unidentifiable': 'unidentifiable',
         'zooplankton': 'swimmer', 'zooplankton_part': 'swimmer'}
    
    return d


def commonize_model_labels():
    
    d = {'aggregate': 'aggregate', 'bubble': 'unidentifiable',
         'fiber_blur': 'fiber', 'fiber_sharp': 'fiber',
         'long_pellet': 'long_pellet', 'mini_pellet': 'mini_pellet',
         'noise': 'unidentifiable', 'phyto_long': 'phytoplankton',
         'phyto_round': 'phytoplankton', 'phyto_dino': 'phytoplankton',
         'rhizaria': 'rhizaria', 'salp_pellet': 'salp_pellet',
         'short_pellet': 'short_pellet', 'none': 'unidentifiable',
         'swimmer': 'swimmer'}

    return d


def add_identity(axes, *line_args, **line_kwargs):
# https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


def calculate_fluxes(cfg):

    metadata = tools.load_metadata(cfg)
    predictions = pd.read_csv('../results/unlabeled_predictions.csv')
    df = metadata.merge(predictions, how='left', on='filename')
    model_labels = commonize_model_labels()
    manual_labels = commonize_manual_labels()

    df = df.loc[df['ESD'].notnull()].copy()  # 23 filenames are in the data folder but not in the metadata
    df['subdir_common'] = df['subdir'].apply(lambda x: model_labels[x])
    
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    axs[0].set_xlabel('Original')
    axs[0].set_ylabel('Predicted')
    axs[1].set_xlabel('Measured')
    axs[2].set_xlabel('Measured')
            
    for s in df['sample'].unique():
        sdf = df.loc[(df['sample'] == s)].copy()
        meas_flux = sdf['measured_flux'].unique()[0]
        meas_flux_e = sdf['measured_flux_u'].unique()[0]
        pred_fluxes = []
        labeled = sdf.loc[sdf['subdir'] != 'none'].copy()
        if len(labeled) > 0:  # JC65 has no labeled images
            pred_flux_labeled = labeled.apply(lambda row: flux_equations(row, 'subdir_common'), axis=1).sum()
        else:
            pred_flux_labeled = 0
        unlabeled = sdf.loc[sdf['subdir'] == 'none'].copy()
        pred_columns = [c for c in sdf.columns if 'prediction_' in c]
        for col in pred_columns:
            unlabeled[f'{col}_common'] = unlabeled[col].apply(lambda x: model_labels[x])
            pred_flux_unlabeled = unlabeled.apply(lambda row: flux_equations(row, f'{col}_common'), axis=1).sum()
            pred_fluxes.append(pred_flux_labeled + pred_flux_unlabeled)
        pred_flux = np.mean(pred_fluxes)
        pred_flux_e = np.std(pred_fluxes, ddof=1)
        
        color = get_domain_color(sdf['domain'].unique()[0])
        axs[1].errorbar(meas_flux, pred_flux, xerr=meas_flux_e, yerr=pred_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
        axs[2].errorbar(meas_flux, pred_flux, xerr=meas_flux_e, yerr=pred_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
        axs[2].set_yscale('log')
        axs[2].set_xscale('log')
        
        if sdf['domain'].unique()[0] in ('RR', 'FK'):
            sdf['orig_label_common'] = sdf['orig_label'].apply(lambda x: manual_labels[x])
            orig_flux = sdf.apply(lambda row: flux_equations(row, 'orig_label_common'), axis=1).sum()
            axs[0].errorbar(orig_flux, pred_flux, yerr=pred_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)

    for ax in axs:
        add_identity(ax, color=black, ls='--')

    lines = [Line2D([0], [0], color=orange, lw=4),
             Line2D([0], [0], color=vermillion, lw=4),
             Line2D([0], [0], color=blue, lw=4),
             Line2D([0], [0], color=green, lw=4)]
    labels = ['FK', 'JC', 'RR', 'SR']
    axs[0].legend(lines, labels, frameon=False, handlelength=1)

    fig.savefig(f'../results/flux_comparison.png', bbox_inches='tight')
        
    
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

    # distribution_barplot(cfg, cfg['ablation_classes'])
    # distribution_barplot(cfg, cfg['ablation_classes'], True)
    # distribution_heatmap(cfg, cfg['ablation_classes'], 'braycurtis', True)
    # prediction_subplots_bar(cfg, cfg['ablation_classes'], ablation_predictions)
    # prediction_subplots_scatter(cfg, cfg['ablation_classes'], ablation_predictions)
    # uniform_comparison_barplots(cfg, ablation_predictions, uniform_predictions)
    calculate_fluxes(cfg)
