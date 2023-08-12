import argparse
import os
import sys

import cartopy
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from itertools import product
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

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
              4: (('SR', 'FK', 'JC', 'SR_FK', 'SR_JC', 'FK_JC', 'SR_FK_JC'), 'RR'),
              5: (('RR', 'FK', 'JC', 'RR_FK', 'RR_JC', 'FK_JC', 'RR_FK_JC'), 'SR'),
              6: (('RR', 'SR', 'JC', 'RR_SR', 'RR_JC', 'SR_JC', 'RR_SR_JC'), 'FK'),
              7: (('RR', 'SR', 'FK', 'RR_SR', 'RR_FK', 'SR_FK', 'RR_SR_FK'), 'JC')}
    
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

    prediction_results = tools.load_json(os.path.join(
        '..', 'results', prediction_results_fname))

    ind = np.arange(len(prediction_results['taa']))
    width = 0.4

    fig, axs = plt.subplots(2, 4, figsize=(13, 6))
    fig.subplots_adjust(left=0.08, hspace=0.5, wspace=0.1)
    fig.supylabel('Accuracy', fontsize=14)
    
    exp_id_dict = get_exp_ids(cfg)

    for i, ax in enumerate(axs.flatten()):
        exp_ids = exp_id_dict[i]
        ind = np.arange(len(exp_ids))
        ax.set_xticks(ind + width, get_ticklabels(exp_ids))
        ax.grid(visible=True, which='major', axis='y', zorder=1)
        taa = [prediction_results['taa'][j] for j in exp_ids]
        tas = [prediction_results['tas'][j] for j in exp_ids]
        ax.bar(ind + width, taa,  width, yerr=tas, color=blue, error_kw={'elinewidth': 1}, zorder=10)
        ax.set_ylim((0.2, 1))
        if i not in (0, 4):
            ax.set_yticklabels([])
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SR', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    axs.flatten()[3].set_title('JC', fontsize=14)

    plt.savefig(os.path.join('..', 'results/prediction_subplots_bar.pdf'), bbox_inches='tight')
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
    classes = cfg['ablation_classes']

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
        ts = [train_sizes[j] for j in exp_ids]
        ax.errorbar(ts, taa, tas, c=blue, fmt='o', ecolor=black)
        ax.set_xlim((0, 17000))
        if i < 4:
            ax.set_xticklabels([])
            ax.set_ylim((0.8, 1))
        else:
            ax.set_ylim((0.3, 1))
        if i not in (0, 4):
            ax.set_yticklabels([])

    axs.flatten()[0].set_ylabel('Accuracy (in-domain)', labelpad=10)
    axs.flatten()[4].set_ylabel('Accuracy (out-of-domain)', labelpad=10)
    fig.text(0.5, 0.04, 'Training size', ha='center')      
                    
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SR', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    axs.flatten()[3].set_title('JC', fontsize=14)
    
    plt.savefig(os.path.join('..', 'results', 'prediction_subplots_scatter_trainsize.pdf'), bbox_inches='tight')
    plt.close()

    # dissimilarity
    df = distribution_heatmap(cfg, classes, 'braycurtis', False)
    fig, axs = plt.subplots(2, 4, figsize=(8, 6))
    fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axs.flatten()):
        exp_ids = exp_id_dict[i]
        train_domains, test_domain = get_df_domains(exp_ids)
        distances = df.loc[test_domain][train_domains]
        taa = [prediction_results['taa'][j] for j in exp_ids]
        tas = [prediction_results['tas'][j] for j in exp_ids]
        ts = [train_sizes[j] for j in exp_ids]
        ax.errorbar(distances, taa, tas, c=blue, fmt='o', ecolor=black)
        ax.set_xlim((-0.05, 0.6))
        if i < 4:
            ax.set_xticklabels([])
            ax.set_ylim((0.8, 1))
        else:
            ax.set_ylim((0.3, 1))
        if i not in (0, 4):
            ax.set_yticklabels([])

    
    axs.flatten()[0].set_ylabel('Accuracy (in-domain)', labelpad=10)
    axs.flatten()[4].set_ylabel('Accuracy (out-of-domain)', labelpad=10)
    fig.text(0.5, 0.04, 'Bray-Curtis dissimilarity', ha='center')
    
    axs.flatten()[0].set_title('RR', fontsize=14)
    axs.flatten()[1].set_title('SR', fontsize=14)
    axs.flatten()[2].set_title('FK', fontsize=14)
    axs.flatten()[3].set_title('JC', fontsize=14)

    plt.savefig(os.path.join('..', 'results', 'prediction_subplots_scatter_BCdis.pdf'), bbox_inches='tight')
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
            split_filepaths = domain_splits[d]['train'] + domain_splits[d]['val'] + domain_splits[d]['test']
            counts_by_label.append(sum(c in f for f in split_filepaths))
        df_list.append(pd.DataFrame(counts_by_label, index=classes, columns=[d]))
    df = pd.concat(df_list, axis=1)

    for i in split_ids:  # for splits that have more than one domain
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


def distribution_barplot(cfg):
    
    classes = ['aggregate', 'long_pellet', 'mini_pellet', 'short_pellet',  'phyto_dino', 'phyto_long',
               'phyto_round', 'rhizaria', 'salp_pellet', 'noise', 'swimmer', 'bubble', 'fiber_blur', 'fiber_sharp',]
    
    ind = np.arange(len(classes)) 
    width = 0.15
    
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), tight_layout=True)
    fig.subplots_adjust(bottom=0.2)
    
    for i, normalize in enumerate((False, True)):
        
        df = get_class_count_df(cfg, classes, normalize=normalize)

        axs[i].bar(ind-width*2, df['FC'], width, color=orange)
        axs[i].bar(ind-width*1, df['FO'], width, color=sky)
        axs[i].bar(ind, df['JC'], width, color=vermillion)
        axs[i].bar(ind+width*1, df['RR'], width, color=blue)
        axs[i].bar(ind+width*2, df['SR'], width, color=green)
        axs[i].set_xticks(ind, df.index.values, rotation=45, ha='right')
        axs[i].axvline(8.42, c=black, ls=':')
        
        if i == 0:
            axs[i].set_ylabel('Number of observations', fontsize=12)
            axs[i].set_yscale('log')
            axs[i].tick_params(labelbottom=False) 
        else:
            axs[i].set_ylabel('Fraction of observations', fontsize=12)

    lines = [Line2D([0], [0], color=orange, lw=6),
             Line2D([0], [0], color=sky, lw=6),
             Line2D([0], [0], color=vermillion, lw=6),
             Line2D([0], [0], color=blue, lw=6),
             Line2D([0], [0], color=green, lw=6)]
    labels = ['FC', 'FO', 'JC', 'RR', 'SR']
    axs[0].legend(lines, labels, ncol=6, bbox_to_anchor=(0.5, 1.02), loc='lower center',
              frameon=False, handlelength=1)
        
    plt.savefig(os.path.join('..', 'results', f'distribution_barplot.pdf'), bbox_inches='tight')
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
    elif domain  == 'JC':
        c = vermillion
    elif domain == 'RR':
        c = blue
    elif domain == 'FC':
        c = orange
    else:
        c = sky
    
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


def flux_equations(row, label_col):

    p_class = row[label_col]
    esd = row['esd']  # µm
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
    elif p_class in ('unidentifiable', 'fiber', 'swimmer'):
        return 0
    else:
        print('UNKNOWN PARTICLE TYPE')
        sys.exit()
    
    return shape_to_flux(a, b, v, area, time)


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


def calculate_flux_df(cfg):
    
    d = {'bubble': 'unidentifiable', 'noise': 'unidentifiable',
         'phyto_long': 'phytoplankton', 'phyto_round': 'phytoplankton', 'phyto_dino': 'phytoplankton',
         'fiber_blur': 'fiber', 'fiber_sharp': 'fiber'}
    
    def row_flux(row):

        if row['label'] != 'none':
            if row['label'] in d:
                row['label_group'] = d[row['label']]
            else:
                row['label_group'] = row['label']
            row['label_flux'] = flux_equations(row, 'label_group')
        else:
            for c in pred_columns:
                if row[c] in d:
                    row[f'{c}_group'] = d[row[c]]
                else:
                    row[f'{c}_group'] = row[c]
                row[f'{c}_flux'] = flux_equations(row, f'{c}_group')
        if row['domain'] in ('RR', 'FK'):
            row['olabel_flux'] = flux_equations(row, 'olabel_group')
        
        return row

    metadata = tools.load_metadata(cfg)
    predictions = pd.read_csv('../results/unlabeled_predictions.csv')
    df = metadata.merge(predictions, how='left', on='filename')
    
    df = df.loc[df['esd'].notnull()].copy()  # 23 filenames are in the data folder but not in the metadata
    pred_columns = [c for c in df.columns if 'pred' in c]
    tqdm.pandas()
    df = df.progress_apply(row_flux, axis=1)
    
    df.to_csv('../results/fluxes.csv', index=False)


def flux_comparison():
    
    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    fig.subplots_adjust(left=0.1, wspace=0.2)
    axs = axs.flatten()
    
    fig.supylabel('Model flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    axs[2].set_xlabel('Original flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    axs[3].set_xlabel('Measured flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    
    df = pd.read_csv('../results/fluxes.csv', index_col=False, low_memory=False)

    all_pred = []
    all_meas = []
    rrfk_pred = []
    rrfk_orig = []
            
    for s in df['sample'].unique():
        
        # if s[0] == 'J' and int(s[2:]) >= 49:
        #     continue

        sdf = df.loc[(df['sample'] == s)].copy()
        meas_flux = sdf['measured_flux'].unique()[0]
        meas_flux_e = sdf['measured_flux_e'].unique()[0]
        
        pred_columns = [c for c in df.columns if 'pred' in c and 'flux' in c]
        pred_fluxes = [sdf['label_flux'].sum() + sdf[c].sum() for c in pred_columns]
            
        pred_flux = np.mean(pred_fluxes)
        pred_flux_e = np.std(pred_fluxes, ddof=1)
        
        color = get_domain_color(sdf['domain'].unique()[0])
        axs[1].errorbar(meas_flux, pred_flux, xerr=meas_flux_e, yerr=pred_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
        axs[3].errorbar(meas_flux, pred_flux, xerr=meas_flux_e, yerr=pred_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)

        all_pred.append(pred_flux)
        all_meas.append(meas_flux)
        
        if sdf['domain'].unique()[0] in ('RR', 'FK'):
            orig_flux = sdf['olabel_flux'].sum()
            rrfk_pred.append(pred_flux)
            rrfk_orig.append(orig_flux)
            axs[0].errorbar(orig_flux, pred_flux, yerr=pred_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
            axs[2].errorbar(orig_flux, pred_flux, yerr=pred_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)

    for i, ax in enumerate(axs):
        add_identity(ax, color=black, ls='--')
        if i > 1:
            ax.set_yscale('log')
            ax.set_xscale('log')
            

    lines = [Line2D([0], [0], color=orange, lw=4),
             Line2D([0], [0], color=vermillion, lw=4),
             Line2D([0], [0], color=blue, lw=4),
             Line2D([0], [0], color=green, lw=4)]
    labels = ['FK', 'JC', 'RR', 'SR']
    axs[0].legend(lines, labels, frameon=False, handlelength=1)

    fig.savefig(f'../results/flux_comparison.pdf', bbox_inches='tight')

    print(np.sqrt(mean_squared_error(rrfk_orig, rrfk_pred)))
    print(np.sqrt(mean_squared_error(all_meas, all_pred)))


def flux_comparison_by_class():
    
    classes = ('aggregate', 'long_pellet', 'short_pellet', 'mini_pellet', 'salp_pellet', 'rhizaria', 'phytoplankton')
    
    fig, axs = plt.subplots(2, 4, figsize=(12,6))
    fig.subplots_adjust(left=0.08, wspace=0.3)
    axs = axs.flatten()
    axs[-1].set_visible(False)
    fig.supxlabel('Original flux (mmol m$^{-2}$ d$^{-1}$)')
    fig.supylabel('Model flux (mmol m$^{-2}$ d$^{-1}$)')
    
    df = pd.read_csv('../results/fluxes.csv', index_col=False, low_memory=False)
    df = df.loc[df['olabel_group'].notnull()]
    pred_columns = [c for c in df.columns if '_' not in c and 'pred' in c]
            
    for s in df['sample'].unique():

        sdf = df.loc[(df['sample'] == s)].copy()
        color = get_domain_color(sdf['domain'].unique()[0])
        
        for i, clss in enumerate(classes):
 
            pred_fluxes = [sdf[sdf['label_group'] == clss]['label_flux'].sum()
                           + sdf[sdf[f'{c}_group'] == clss][f'{c}_flux'].sum() for c in pred_columns]
            pred_flux = np.mean(pred_fluxes)
            pred_flux_e = np.std(pred_fluxes, ddof=1)
            
            axs[i].errorbar(sdf[sdf['olabel_group'].str.contains(clss)]['olabel_flux'].sum(), pred_flux, yerr=pred_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
            
    for i, clss in enumerate(classes):
        add_identity(axs[i], color=black, ls='--')
        axs[i].text(0.98, 0.02, clss, ha='right', va='bottom', size=10, transform=transforms.blended_transform_factory(axs[i].transAxes, axs[i].transAxes))

    lines = [Line2D([0], [0], color=orange, lw=4),
             Line2D([0], [0], color=blue, lw=4)]
    labels = ['FK', 'RR']
    axs[0].legend(lines, labels, frameon=False, handlelength=1)

    fig.savefig(f'../results/flux_comparison_byclass.pdf', bbox_inches='tight')


def agreement_rates():
    
    def compare_cols(df, col1, col2):
        
        n_matches = len(df[df[col1] == df[col2]])
        relabel_rate = n_matches / len(df) * 100
        
        return relabel_rate
    
    flux_classes = ['aggregate', 'long_pellet', 'short_pellet', 'mini_pellet', 'salp_pellet', 'rhizaria', 'phytoplankton']
    all_classes = flux_classes + ['fiber', 'swimmer', 'unidentifiable']
        
    # load df
    df = pd.read_csv('../results/fluxes.csv', index_col=False, low_memory=False)
    df = df.loc[df['olabel'].notnull()]
    
    # group model labels
    pred_cols = [c for c in df.columns if 'pred' in c and 'group' in c]
    
    # Original vs. relabeled
    relabeled = df.loc[df['relabel_group'].notnull()]
    r = compare_cols(relabeled, 'olabel_group', 'relabel_group')
    print(f'Original, relabeled: {r:.2f}')
    
    print('-----------------')
    
    # Unambiguous comparisons
    unambig = df.loc[df['label'] != 'none']
    r = compare_cols(unambig, 'olabel_group', 'label_group')
    print(f'Original, labeled unambig: {r:.2f}')
    
    unambig_relabeled = unambig.loc[unambig['relabel_group'].notnull()]
    r = compare_cols(unambig_relabeled, 'relabel_group', 'label_group')
    print(f'Relabeled, labeled unambig: {r:.2f}')
    
    r = compare_cols(unambig_relabeled, 'olabel_group', 'relabel_group')
    print(f'Original, relabeled (both unambig): {r:.2f}')
    
    print('-----------------')

    # Ambiguous comparisons
    ambig = df.loc[df['label'] == 'none']
    r = []
    for c in pred_cols:
        r.append(compare_cols(ambig, 'olabel_group', c))
    print(f'Original, predicted ambig: {np.mean(r):.2f} ± {np.std(r, ddof=1):.2f}')
    
    ambig_relabeled = ambig.loc[ambig['relabel_group'].notnull()]
    r = []
    for c in pred_cols:
        r.append(compare_cols(ambig_relabeled, 'relabel_group', c))
    print(f'Relabeled, predicted ambig: {np.mean(r):.2f} ± {np.std(r, ddof=1):.2f}')
    
    r = compare_cols(ambig_relabeled, 'olabel_group', 'relabel_group')
    print(f'Original, relabeled (both ambig): {r:.2f}')
    
    print('-----------------')
    
    fig, axs = plt.subplots(3, 2, figsize=(16, 20))
    fig.subplots_adjust(hspace=0.5)
    axs = axs.flatten()
    axs[-1].set_visible(False)

    for c in pred_cols:
        i = c.split('_')[0][-1]
        ax = axs[int(i)]
        t = ambig[['olabel_group', c]]
        t = t[t.isin(flux_classes).any(axis=1)]
        cm = ConfusionMatrixDisplay.from_predictions(t['olabel_group'], t[c], ax=ax, cmap=plt.cm.Greens, xticks_rotation=90, labels=all_classes, colorbar=False)
        if i in ('0', '2', '4'):
            ax.set_ylabel('Original')
        else:
            ax.set_ylabel('')
        ax.set_xlabel(f'Model replicate {int(i) + 1}')
        # https://stackoverflow.com/questions/66483409/adjust-size-of-confusionmatrixdisplay-scikitlearn
        cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.01, ax.get_position().height])
        plt.colorbar(cm.im_,  cax=cax, ax=ax)
        # ax.axhline(len(flux_classes) - 0.5, color=black)
        # ax.axvline(len(flux_classes) - 0.5, color=black)
    fig.savefig(f'../results/cmatrices.pdf', bbox_inches='tight')
    

def print_image_counts():

    metadata = tools.load_metadata(cfg)
    for domain in metadata['domain'].unique():
        df = metadata.loc[metadata['domain'] == domain]
        labeled = df.loc[df['label'] != 'none']
        n_labeled = len(labeled)
        percent_labeled = n_labeled/len(df) * 100
        print(f'{domain}: {len(df)} images, {n_labeled} labeled ({percent_labeled:.0f}%)')


def draw_map():

    lats = (22.3, 27.7, 34.7, 50, 49, 34.3)
    lons = (-151.9, -139.5, -123.5, -145, -15, -120)
    text_lats = (25, 34.7, 53, 52, 30.3)
    text_lons = (-145.7, -128.5, -145, -15, -120)
    domains = ['FO', 'FO', 'FC', 'RR', 'JC', 'SR']
    cols = [get_domain_color(d) for d in domains]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=cartopy.crs.PlateCarree())
    ax.add_feature(cartopy.feature.LAND.with_scale('110m'), color='k', zorder=2)
    ax.set_extent([-160, 0, 10, 70], crs=cartopy.crs.PlateCarree())
    gl = ax.gridlines(draw_labels=['left', 'bottom'], zorder=1)
    # gl.xlines = False

    ax.scatter(lons, lats, color=cols, s=30, zorder=3)
    for i, d in enumerate(domains[1:]):
        ax.text(text_lons[i], text_lats[i], d, va='center', ha='center')
        
    plt.savefig('../results/map.pdf', bbox_inches='tight')
    plt.close()

def esd_by_class(cfg):

    df = tools.load_metadata(cfg)[['label', 'esd', 'domain']]
    df = df.loc[(df['esd'].notnull() & df['label'] != 'none')]
    # df['color'] = df.apply(lambda x: get_domain_color(x['domain']), axis=1)

    classes = cfg['all_classes']
    median_esds = [df.loc[df['label'] == c]['esd'].median() for c in classes]
    classes = [x for _, x in sorted(zip(median_esds, classes))]
    median_esds.sort()

    fig, axs = plt.subplots(3, 5, tight_layout=True, figsize=(14,6))
    axs = axs.flatten()
    axs[-1].axis('off')
    fig.supxlabel('ESD (µm)')
    fig.supylabel('Frequency')

    for i, c in enumerate(classes):
        c_df = df[df['label'] == c]
        axs[i].hist(c_df['esd'])
        axs[i].axvline(median_esds[i], color=black, ls='--')
        axs[i].text(0.98, 0.98, c, ha='right', va='top', size=10, transform=transforms.blended_transform_factory(axs[i].transAxes, axs[i].transAxes))
        axs[i].text(0.98, 0.84, f'med: {median_esds[i]:.0f}', ha='right', va='top', size=10, transform=transforms.blended_transform_factory(axs[i].transAxes, axs[i].transAxes))
        axs[i].text(0.98, 0.70, f'min: {c_df["esd"].min():.0f}', ha='right', va='top', size=10, transform=transforms.blended_transform_factory(axs[i].transAxes, axs[i].transAxes))
        axs[i].text(0.98, 0.56, f'max: {c_df["esd"].max():.0f}', ha='right', va='top', size=10, transform=transforms.blended_transform_factory(axs[i].transAxes, axs[i].transAxes))

    plt.savefig('../results/esd_by_class.pdf', bbox_inches='tight')
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

    distribution_barplot(cfg)
    # prediction_subplots_bar(cfg, ablation_predictions)
    # prediction_subplots_scatter(cfg, ablation_predictions)
    # uniform_comparison_barplots(cfg, ablation_predictions, uniform_predictions)
    # calculate_flux_df(cfg)
    # flux_comparison()
    # flux_comparison_by_class()
    # agreement_rates()
    draw_map()
    esd_by_class(cfg)
    # print_image_counts()
