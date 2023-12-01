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
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, classification_report
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import src.dataset as dataset
import src.tools as tools


def training_plots():

    models = [f for f in os.listdir(f'../results/weights') if f'.pt' in f]

    for m in models:
        exp_id = m.split('.')[0]
        model_output = torch.load(f'../results/weights/{m}', map_location='cpu')
        num_epochs = len(model_output['train_loss_hist'])

        plt.xlabel('Training Epochs')
        plt.ylabel('Accuracy')
        plt.plot(range(1, num_epochs + 1), model_output['train_acc_hist'],
                 label='training')
        plt.plot(range(1, num_epochs + 1), model_output['val_acc_hist'],
                 label='validation')
        plt.legend()
        plt.savefig(f'../results/figs/accuracy_{exp_id}.png')
        plt.close()

        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.plot(range(1, num_epochs + 1), model_output['train_loss_hist'],
                 label='training')
        plt.plot(range(1, num_epochs + 1), model_output['val_loss_hist'],
                 label='validation')
        plt.legend()
        plt.savefig(f'../results/figs/loss_{exp_id}.png')
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
        print(label_col)
        print(row)
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


def calculate_flux_df(domain=None):
    
    d = {'bubble': 'unidentifiable', 'noise': 'unidentifiable',
         'phyto_long': 'phytoplankton', 'phyto_round': 'phytoplankton', 'phyto_dino': 'phytoplankton',
         'fiber_blur': 'fiber', 'fiber_sharp': 'fiber'}
    
    def row_flux(row):

        for c in cols_to_group:
            if row[c] != 'none' and row[c]==row[c]:  # second statement deals with NaNs
                if row[c] in d:
                    row[f'{c}_group'] = d[row[c]]
                else:
                    row[f'{c}_group'] = row[c]
                row[f'{c}_flux'] = flux_equations(row, f'{c}_group')

        if row['domain'] in ('RR', 'JC'):
            row['olabel_flux'] = flux_equations(row, 'olabel_group')
        
        return row

    df_list = []
    csvs = [f for f in os.listdir('../results/predictions') if f'target{domain}' in f]
    for f in csvs:
        model = f.split(".")[0]
        pred_df = pd.read_csv(f'../results/predictions/{f}', header=0)[['filepath', 'prediction']]
        pred_df['filename'] = pred_df['filepath'].str.split('/', expand=True)[1]
        pred_df = pred_df.drop('filepath', axis=1).set_index('filename')
        pred_df = pred_df.rename(columns={'prediction': f'prediction{model}'})
        if 'ood' not in model:
            splits_file = f'{model.split("-")[0]}.json'
            splits_dict = tools.load_json(f'../data/{splits_file}')
            train_fps = [f for f in splits_dict['train'] if domain in f]
            val_fps = [f for f in splits_dict['val'] if domain in f]
            train_df = pd.DataFrame(train_fps + val_fps, columns=['filepath'])
            train_df[[f'label{model}','filename']] = train_df['filepath'].str.split('/', expand=True)
            train_df = train_df.drop('filepath', axis=1).set_index('filename')
            df_list.extend([pred_df, train_df])
        else:
            df_list.append(pred_df)
    temp_df = pd.concat(df_list, axis=1)  # outer join by default

    metadata = tools.load_metadata()
    if domain is not None:
        metadata = metadata.loc[metadata['domain']==domain]
    df = metadata.merge(temp_df, how='left', on='filename')
    df = df.loc[df['esd'].notnull()].copy()  # 23 filenames are in the data folder but not in the metadata
    cols_to_group = [c for c in df.columns if '-' in c] + ['label']
    tqdm.pandas()
    df = df.progress_apply(row_flux, axis=1)
    # df = df.apply(row_flux, axis=1)
    
    df.to_csv(f'../results/fluxes/{domain}.csv', index=False)


def flux_comparison_by_class(domain):
    
    classes = ['aggregate', 'long_pellet', 'short_pellet', 'mini_pellet', 'salp_pellet', 'rhizaria', 'phytoplankton']
    
    df = pd.read_csv(f'../results/fluxes/{domain}.csv', index_col=False, low_memory=False)
    samples = df['sample'].unique()
    if domain == 'JC':
        samples = [s for s in samples if int(s[2:]) < 49]  # samples above 49 were problematic for flux estimation
    models = (f'target{domain}_ood', f'target{domain}_top1k', f'target{domain}_verify', f'target{domain}_minboost')
    replicates = 5
    flux_dict = {}

    for m in models:

        fig, axs = plt.subplots(3, 3, figsize=(10,10), tight_layout=True)
        fig.subplots_adjust(wspace=0.3)
        axs = axs.flatten()

        flux_dict[m] = {}
            
        for s in samples:

            sdf = df.loc[(df['sample'] == s)].copy()
            color = get_domain_color(sdf['domain'].unique()[0])
            flux_dict[m][s] = {c: {} for c in classes}
            flux_dict[m][s]['human_total'] = sdf['olabel_flux'].sum()
            flux_dict[m][s]['model_total'] = np.zeros(replicates)
            flux_dict[m][s]['measured'] = sdf['measured_flux'].unique()[0]
            flux_dict[m][s]['measured_e'] = sdf['measured_flux_e'].unique()[0]
            
            for i, c in enumerate(classes):
                
                class_flux_human = sdf[sdf['olabel_group'] == c]['olabel_flux'].sum()
                flux_dict[m][s][c]['human'] = class_flux_human
                flux_dict[m][s][c]['model'] = np.zeros(replicates)

                for j in range(replicates):

                    pred_flux = sdf[sdf[f'prediction{m}-{j}_group'] == c][f'prediction{m}-{j}_flux'].sum()
                    if 'ood' not in m:
                        train_flux = sdf[sdf[f'label{m}-{j}_group'] == c][f'label{m}-{j}_flux'].sum()
                        class_flux = train_flux + pred_flux
                    else:
                        class_flux = pred_flux
                    flux_dict[m][s][c]['model'][j] = class_flux
                    flux_dict[m][s]['model_total'][j] += class_flux

                class_flux_mean = np.mean(flux_dict[m][s][c]['model'])
                class_flux_e = np.std(flux_dict[m][s][c]['model'], ddof=1)
                
                axs[i].errorbar(class_flux_human, class_flux_mean, yerr=class_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)

            total_flux_mean = np.mean([flux_dict[m][s]['model_total'][j] for j in range(replicates)])
            total_flux_e = np.std([flux_dict[m][s]['model_total'][j] for j in range(replicates)], ddof=1)
            axs[7].errorbar(flux_dict[m][s]['human_total'], total_flux_mean, yerr=total_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
            axs[8].errorbar(flux_dict[m][s]['measured'], total_flux_mean, xerr=flux_dict[m][s]['measured_e'], yerr=total_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)


        for i, a in enumerate(axs):
            add_identity(a, color=black, ls='--')
            if i < 7:
                text_str = classes[i]        
            elif i == 7:
                text_str = 'total'
            else:
                text_str = 'measured'
            a.text(0.98, 0.02, text_str, ha='right', va='bottom', size=10, transform=transforms.blended_transform_factory(axs[i].transAxes, axs[i].transAxes))


        fig.supxlabel('Human flux (mmol m$^{-2}$ d$^{-1}$)')
        fig.supylabel('Model flux (mmol m$^{-2}$ d$^{-1}$)')
        fig.savefig(f'../results/figs/flux_comparison_byclass_{m}.{image_format}', bbox_inches='tight')
        plt.close()

    x_vars = classes + ['total', 'measured']
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(10,5))
    ax.set_xticks(range(len(x_vars)), labels=x_vars, rotation=45)
    colors = [black, blue, green, orange, vermillion, radish, sky]
    markers = ['o', '^', '+', 's', 'd', 'x', '*']

    def plot_mae(i, k, maes):

        offset = (-0.15, -0.05, 0.05, 0.15)
        mae = np.mean(maes)
        mae_e = np.std(maes, ddof=1)
        print(f'{m}: {mae:.2f} ± {mae_e:.2f}')
        ax.errorbar(i + offset[k], mae, yerr=mae_e, marker=markers[k], c=colors[k], capsize=2)


    i = 0
    with open(f'../results/figs/flux_mae_{domain}.txt', 'w') as sys.stdout:
        print('******FLUX MAEss******')
        for c in classes:
            print(f'---{c}---')
            for k, m in enumerate(models):
                human_class_fluxes = [flux_dict[m][s][c]['human'] for s in samples]
                model_class_fluxes = [[flux_dict[m][s][c]['model'][j] for s in samples] for j in range(replicates)]
                maes = [mean_absolute_error(human_class_fluxes, model_class_fluxes[j]) for j in range(replicates)]
                plot_mae(i, k, maes)
            i += 1
        print(f'---total---')
        for k, m in enumerate(models):
            human_total_fluxes = [flux_dict[m][s]['human_total'] for s in samples]
            model_total_fluxes = [[flux_dict[m][s]['model_total'][j] for s in samples] for j in range(replicates)]
            maes = [mean_absolute_error(human_total_fluxes, model_total_fluxes[j]) for j in range(replicates)]
            plot_mae(i, k, maes)
        i += 1
        print(f'---measured---')
        for k, m in enumerate(models):
            measured_fluxes = [flux_dict[m][s]['measured'] for s in samples]
            model_total_fluxes = [[flux_dict[m][s]['model_total'][j] for s in samples] for j in range(replicates)]
            maes = [mean_absolute_error(measured_fluxes, model_total_fluxes[j]) for j in range(replicates)]
            plot_mae(i, k, maes)

    legend_text = ('OOD', '+top1k', '+verify', '+minboost')
    lines = [Line2D([0], [0], color=colors[z], lw=6) for z, _ in enumerate(models)]
    ax.legend(lines, legend_text, ncol=len(models), bbox_to_anchor=(0.5, 1.02), loc='lower center',
            frameon=False, handlelength=1)

    human_measured_mae = flux_comparison_human_measured(domain)
    ax.hlines(human_measured_mae, 7.6, 8.4, colors=black, alpha=0.3)
    
    fig.supylabel('MAE (mmol m$^{-2}$ d$^{-1}$)')

    fig.savefig(f'../results/figs/flux_comparison_mae_{domain}.{image_format}', bbox_inches='tight')
    plt.close()


def compare_accuracies(domain):
    
    def compare_cols(df, col1, col2):
        
        n_matches = len(df[df[col1] == df[col2]])
        accuracy = n_matches / len(df) * 100
        
        return accuracy
            
    df = pd.read_csv('../results/fluxes/{domain}.csv', index_col=False)
    df = df.loc[df['olabel'].notnull()]
    
    pred_col = 'prediction0_group'
    
    unambig = df.loc[df['label'] != 'none']
    r = compare_cols(unambig, 'label_group', pred_col)
    print(f'perfect labels vs. predictions: {r:.2f}')
    
    r = compare_cols(df, 'olabel_group', pred_col)
    print(f'perfect + shady labels vs. predictions: {r:.2f}')
    
    relabeled = df.loc[df['relabel_group'].notnull()]
    r = compare_cols(relabeled, 'olabel_group', 'relabel_group')
    print(f'Intra-annotator agreement (perfect + shady labels): {r:.2f}')
    

def print_image_counts():

    metadata = tools.load_metadata()
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

    classes = cfg['classes']
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

    plt.savefig('../runs/esd_by_class.png', bbox_inches='tight')
    plt.close()

    fig.savefig(f'../results/figs/pad_exp_metrics.png')


def metrics_hptune():

    open(f'../results/figs/metrics_hptune.txt', 'w')  # delete metrics file if it exists

    exp_dict = {'preprocessing': ['targetRR_ood', 'pad', 'normdata', 'normIN', 'padnormdata', 'padnormIN'],
                'learningrate': ['targetRR_ood', 'highLR', 'lowLR'],
                'weightdecay': ['targetRR_ood', 'highWD', 'lowWD']}

    labels = yaml.safe_load(open('../configs/targetRR_ood.yaml', 'r'))['classes']
    y_vars = ('precision', 'recall')
    x_vars = labels + ['macro avg', 'weighted avg']
    colors = [blue, green, orange, vermillion, black, radish, sky]
    markers = ['o', '^', '+', 's', 'd', 'x', '*']

    for exp in exp_dict:

        cfg_names = sorted(exp_dict[exp])

        fig, axs = plt.subplots(len(y_vars), 1, tight_layout=True, figsize=(10,5))
        axs[-1].set_xticks(range(len(x_vars)), labels=x_vars, rotation=45)
        prediction_files = [f for f in os.listdir(f'../results/predictions') if f.split('-')[0] in cfg_names]
        reports = {}

        for f in prediction_files:
            df = pd.read_csv(f'../results/predictions/{f}')
            df['label'] = df['filepath'].apply(lambda x: x.split('/')[0])
            df = df.loc[df['label'] != 'none']
            cm, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
            ConfusionMatrixDisplay.from_predictions(
                df['label'],
                df['prediction'],
                cmap=plt.cm.Blues,
                normalize=None,
                xticks_rotation='vertical',
                values_format='.0f',
                ax=ax,
                labels=labels)
            cm.savefig(f'../results/figs/cmatrix_{f.split(".")[0]}.png')
            plt.close(cm)
            reports[f] = classification_report(
                df['label'], df['prediction'], output_dict=True, zero_division=0, labels=labels)
        
        for i, y in enumerate(y_vars):
            axs[i].set_ylabel(y)
            if i != len(axs) - 1:
                axs[i].set_xticklabels([])
                axs[i].set_xticks(range(len(x_vars)))
            for j, x in enumerate(x_vars):
                with open(f'../results/figs/metrics_hptune.txt', 'a') as sys.stdout:
                    print(f'----{y}, {x}----')
                    for m, c in enumerate(cfg_names):
                        keys = [k for k in reports.keys() if f'{c}-' in k]
                        y_avg = np.mean([reports[k][x][y] for k in keys])
                        y_std = np.std([reports[k][x][y] for k in keys], ddof=1)
                        axs[i].errorbar(j, y_avg, y_std, color=colors[m], ecolor=colors[m], marker=markers[m], capsize=2)
                        print(f'{c}: {y_avg*100:.2f} ± {y_std*100:.2f}')
        
        lines = [Line2D([0], [0], color=colors[m], lw=6) for m, _ in enumerate(cfg_names)]
        axs[0].legend(lines, cfg_names, ncol=len(cfg_names), bbox_to_anchor=(0.5, 1.02), loc='lower center',
                frameon=False, handlelength=1)      

        fig.savefig(f'../results/figs/metrics_{exp}.{image_format}')
        plt.close(fig)


def metrics_hitloop(domain):

    open(f'../results/figs/metrics_hitloop_{domain}.txt', 'w')  # delete metrics file if it exists
    flux_df = pd.read_csv(f'../results/fluxes/{domain}.csv', index_col='filename', low_memory=False)
    human_df = tools.load_metadata()
    if domain == 'RR':
        relabel_df = human_df.loc[human_df['relabel_group'].notnull()]
        relabel_report = classification_report(relabel_df['olabel_group'], relabel_df['relabel_group'], output_dict=True)
    human_df = human_df.loc[human_df['domain'] == domain][['filename', 'olabel']]
    human_df = human_df.rename(columns={'olabel': f'label'})
    human_df.set_index('filename', inplace=True)
    

    models = (f'target{domain}_ood', f'target{domain}_top1k', f'target{domain}_verify', f'target{domain}_minboost')
    offset = (-0.15, -0.05, 0.05, 0.15)
    replicates = 5
    y_vars = ('precision', 'recall')
    labels = sorted(flux_df['olabel_group'].unique())
    colors = [black, blue, green, orange, vermillion, radish, sky]
    markers = ['o', '^', '+', 's', 'd', 'x', '*']

    fig, axs = plt.subplots(len(y_vars), 1, tight_layout=True, figsize=(10,5))
    axs[-1].set_xticks(range(len(labels)), labels=labels, rotation=45)
    prediction_files = [f'{m}-{i}' for (m, i) in product(models, range(replicates))]
    reports = {}

    for f in prediction_files:

        cols = [c for c in flux_df.columns if f in c and 'group' in c] + ['olabel_group']
        df = flux_df[cols].copy()
        df['prediction'] = df.filter(like=f).ffill(axis=1).iloc[:,-1]  # prediction = label if image was labeled, otherwise prediction
        df.rename(columns={'olabel_group': 'label'}, inplace=True)

        cm, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
        ConfusionMatrixDisplay.from_predictions(
            df['label'],
            df['prediction'],
            cmap=plt.cm.Blues,
            normalize=None,
            xticks_rotation='vertical',
            values_format='.0f',
            ax=ax,
            labels=labels)
        cm.savefig(f'../results/figs/cmatrix_{f}.png')
        plt.close(cm)
        reports[f] = classification_report(
            df['label'], df['prediction'], output_dict=True, zero_division=0, labels=labels)
    
    for i, y in enumerate(y_vars):
        axs[i].set_ylabel(y)
        if i != len(axs) - 1:
            axs[i].set_xticklabels([])
            axs[i].set_xticks(range(len(labels)))
        for j, x in enumerate(labels):
            with open(f'../results/figs/metrics_hitloop_{domain}.txt', 'a') as sys.stdout:
                print(f'----{y}, {x}----')
                for z, m in enumerate(models):
                    keys = [k for k in reports.keys() if f'{m}-' in k]
                    y_avg = np.mean([reports[k][x][y] for k in keys])
                    y_std = np.std([reports[k][x][y] for k in keys], ddof=1)
                    axs[i].errorbar(j + offset[z], y_avg, y_std, color=colors[z], ecolor=colors[z], marker=markers[z], capsize=2)
                    print(f'{m}: {y_avg*100:.2f} ± {y_std*100:.2f}')
                if domain == 'RR':
                    axs[i].hlines(relabel_report[x][y], j + min(offset), j + max(offset), color=black, alpha=0.3)
                    print(f'relabel: {relabel_report[x][y]:.2f}')


    legend_text = ('OOD', '+top1k', '+verify', '+minboost')
    lines = [Line2D([0], [0], color=colors[z], lw=6) for z, _ in enumerate(models)]
    axs[0].legend(lines, legend_text, ncol=len(models), bbox_to_anchor=(0.5, 1.02), loc='lower center',
            frameon=False, handlelength=1)     

    fig.savefig(f'../results/figs/metrics_hitloop_{domain}.{image_format}')
    plt.close(fig)

    
def softmax_histograms(cfg):

    df = pd.read_csv(f'../results/predictions_test.csv')
    df['label'] = df['filepath'].apply(lambda x: os.path.basename(os.path.split(x)[0]))
    df = df.loc[df['label'] != 'none']
    cols = [c for c in df.columns if c in cfg['classes']]

    fig, axs = plt.subplots(4, 4, tight_layout=True, figsize=(10,10))
    axs = axs.flatten()

    for i, c in enumerate(cols):
        c_df = df.loc[df['label'] == c]
        axs[i].hist(c_df[c].values)
        axs[i].set_xlabel(c)
        axs[i].set_xlim(0,1)

    plt.savefig(f'../results/figs/softmax_histogram.png')
    plt.close()


def flux_comparison_human_measured(domain):
    
    fig, ax = plt.subplots(1, 1)
    
    ax.set_xlabel('Measured flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    ax.set_ylabel('Human flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)

    df = pd.read_csv(f'../results/fluxes/{domain}.csv', index_col=False, low_memory=False)

    meas_flux_allsamples = []
    annot_flux_allsamples = []
            
    for s in df['sample'].unique():
        
        if domain == 'JC' and int(s[2:]) >= 49:
            continue

        sdf = df.loc[(df['sample'] == s)].copy()
        meas_flux = sdf['measured_flux'].unique()[0]
        meas_flux_e = sdf['measured_flux_e'].unique()[0]
        annot_flux = sdf['olabel_flux'].sum()

        meas_flux_allsamples.append(meas_flux)
        annot_flux_allsamples.append(annot_flux)
        
        color = get_domain_color(sdf['domain'].unique()[0])
        ax.errorbar(meas_flux, annot_flux, xerr=meas_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
        
    add_identity(ax, color=black, ls='--')
    mae = mean_absolute_error(meas_flux_allsamples, annot_flux_allsamples)
    ax.text(0.98, 0.02, f'MAE: {mae:.2f}', ha='right', va='bottom', size=10, transform=transforms.blended_transform_factory(ax.transAxes, ax.transAxes))

    fig.savefig(f'../results/figs/flux_comparison_human_{domain}.{image_format}', bbox_inches='tight')

    return mae


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
    parser.add_argument('--image_format', '-i', default='png')
    args = parser.parse_args()
    image_format = args.image_format

    training_plots()
    metrics_hptune()
    
    calculate_flux_df('RR')
    flux_comparison_human_measured('RR')
    flux_comparison_by_class('RR')
    metrics_hitloop('RR')

    calculate_flux_df('JC')
    flux_comparison_human_measured('JC')
    flux_comparison_by_class('JC')
    metrics_hitloop('JC')

