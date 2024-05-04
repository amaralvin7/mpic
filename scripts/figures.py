import argparse
import os
import sys

import cartopy
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
import yaml
from itertools import product
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay, mean_absolute_error, classification_report
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

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


def get_class_count_df(classes, normalize=False):

    meta = tools.load_metadata()
    
    df_list = []
    for d in ('FC', 'FO', 'JC', 'RR', 'SR'):  # first, get all of the counts in the individual domains
        counts_by_label = []
        d_df = meta.loc[meta['domain'] == d]
        for c in classes:
            c_df = d_df.loc[d_df['label'] == c]
            counts_by_label.append(len(c_df))
        df_list.append(pd.DataFrame(counts_by_label, index=classes, columns=[d]))
    df = pd.concat(df_list, axis=1)

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


def distribution_barplot():
    
    classes = ['aggregate', 'long_pellet', 'mini_pellet', 'phyto_dino', 'phyto_long',
               'phyto_round', 'rhizaria', 'salp_pellet', 'short_pellet', 'bubble',
               'fiber_blur', 'fiber_sharp', 'noise', 'swimmer']
    
    ind = np.arange(len(classes)) 
    width = 0.15
    
    fig, axs = plt.subplots(2, 1, figsize=(9, 6), tight_layout=True)
    fig.subplots_adjust(bottom=0.2)
    
    for i, normalize in enumerate((False, True)):
        
        df = get_class_count_df(classes, normalize=normalize)

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
        
    plt.savefig(os.path.join('..', 'results', 'figs', f'distribution_barplot.{image_format}'), bbox_inches='tight')
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
            a = 0.113 * 10**-9
            b = 0.81
        elif p_class == 'mini_pellet':
            a = 0.113 * 10**-9
            b = 1
        elif p_class == 'rhizaria':
            a = 0.004 * 10**-9
            b = 0.939
        else:  # phytoplankton
            a = 0.288 * 10**-9
            b = 0.811
    elif p_class == 'long_pellet':
        v = particle_volume('cylinder', esd, (264, 584))
        a = 0.113 * 10**-9
        b = 1
    elif p_class == 'short_pellet':
        v = particle_volume('ellipsoid', esd)
        a = 0.113 * 10**-9
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


def anova_tukey(values, models):

    sig_level = 0.05
    _, anova_pval = scipy.stats.f_oneway(*values)
    print(f'ANOVA p-val: {anova_pval:.6f}')
    if anova_pval < sig_level:
        tukey_results = scipy.stats.tukey_hsd(*values)
        for ((i, j), p) in np.ndenumerate(tukey_results.pvalue):
            if j > i and p < sig_level:  # upper triangle only
                print(f'Tukey p-val ({models[i].split("_")[1]}/{models[j].split("_")[1]}): {p:.6f}')


def flux_comparison_by_class():
    
    def plot_mae(i, k, axs, maes):

        offset = (-0.15, -0.05, 0.05, 0.15)
        mae = np.mean(maes)
        mae_e = np.std(maes, ddof=1)
        print(f'{m}: {mae:.2f} ± {mae_e:.2f}')
        axs[i].errorbar(offset[k], mae, yerr=mae_e, marker=markers[k], c=colors[k], capsize=2)
        axs[i].set_xticks([])
        axs[i].set_xlim(-0.225, 0.225)
        if ii == 1:
            axs[i].set_xlabel(mae_x_vars[i], rotation=45)

    open(f'../results/textfiles/flux_maes.txt', 'w')  # delete flux_maes file if it exists
    classes = ['aggregate', 'long_pellet', 'mini_pellet', 'phytoplankton', 'rhizaria', 'salp_pellet', 'short_pellet']
    mae_x_vars = ['measured', 'total'] + classes
    domains = ('RR', 'JC')
    human_measured_mae = flux_comparison_human_measured()
    flux_dict = {}
    mae_fig, mae_axs = plt.subplots(2, len(mae_x_vars), figsize=(12,5), layout='constrained')

    for ii, d in enumerate(domains):

        mae_axs[ii][0].set_ylabel(d)
        df = pd.read_csv(f'../results/fluxes/{d}.csv', index_col=False, low_memory=False)
        samples = df['sample'].unique()
        if d == 'JC':
            samples = [s for s in samples if int(s[2:]) < 49]  # samples above 49 were problematic for flux estimation
        models = (f'target{d}_ood', f'target{d}_top1k', f'target{d}_verify', f'target{d}_minboost')
        replicates = 5
        flux_dict[d] = {}

        for m in models:

            fig, axs = plt.subplots(3, 3, figsize=(10,10), tight_layout=True)
            fig.subplots_adjust(wspace=0.3)
            axs = axs.flatten()

            flux_dict[d][m] = {}
                
            for s in samples:

                sdf = df.loc[(df['sample'] == s)].copy()
                color = get_domain_color(sdf['domain'].unique()[0])
                flux_dict[d][m][s] = {c: {} for c in classes}
                flux_dict[d][m][s]['human_total'] = sdf['olabel_flux'].sum()
                flux_dict[d][m][s]['model_total'] = np.zeros(replicates)
                flux_dict[d][m][s]['measured'] = sdf['measured_flux'].unique()[0]
                flux_dict[d][m][s]['measured_e'] = sdf['measured_flux_e'].unique()[0]
                
                for i, c in enumerate(classes):
                    
                    class_flux_human = sdf[sdf['olabel_group'] == c]['olabel_flux'].sum()
                    flux_dict[d][m][s][c]['human'] = class_flux_human
                    flux_dict[d][m][s][c]['model'] = np.zeros(replicates)

                    for j in range(replicates):

                        pred_flux = sdf[sdf[f'prediction{m}-{j}_group'] == c][f'prediction{m}-{j}_flux'].sum()
                        if 'ood' not in m:
                            train_flux = sdf[sdf[f'label{m}-{j}_group'] == c][f'label{m}-{j}_flux'].sum()
                            class_flux = train_flux + pred_flux
                        else:
                            class_flux = pred_flux
                        flux_dict[d][m][s][c]['model'][j] = class_flux
                        flux_dict[d][m][s]['model_total'][j] += class_flux

                    class_flux_mean = np.mean(flux_dict[d][m][s][c]['model'])
                    class_flux_e = np.std(flux_dict[d][m][s][c]['model'], ddof=1)
                    
                    axs[i].errorbar(class_flux_human, class_flux_mean, yerr=class_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)

                total_flux_mean = np.mean([flux_dict[d][m][s]['model_total'][j] for j in range(replicates)])
                total_flux_e = np.std([flux_dict[d][m][s]['model_total'][j] for j in range(replicates)], ddof=1)
                axs[7].errorbar(flux_dict[d][m][s]['human_total'], total_flux_mean, yerr=total_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
                axs[8].errorbar(flux_dict[d][m][s]['measured'], total_flux_mean, xerr=flux_dict[d][m][s]['measured_e'], yerr=total_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)


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
            plt.close(fig)

        domain_axs = mae_axs[ii]
        colors = [black, blue, green, orange, vermillion, radish, sky]
        markers = ['o', '^', '+', 's', 'd', 'x', '*']

        i = 0
        with open(f'../results/textfiles/flux_maes.txt', 'a') as sys.stdout:
            print(f'******FLUX MAEs ({d})******')
            print(f'---measured---')
            measured_maes = []
            for k, m in enumerate(models):
                measured_fluxes = [flux_dict[d][m][s]['measured'] for s in samples]
                model_total_fluxes = [[flux_dict[d][m][s]['model_total'][j] for s in samples] for j in range(replicates)]
                maes = [mean_absolute_error(measured_fluxes, model_total_fluxes[j]) for j in range(replicates)]
                plot_mae(i, k, domain_axs, maes)
                measured_maes.append(maes)
            anova_tukey(measured_maes, models)
            i += 1
            print(f'---total---')
            total_maes = []
            for k, m in enumerate(models):
                human_total_fluxes = [flux_dict[d][m][s]['human_total'] for s in samples]
                model_total_fluxes = [[flux_dict[d][m][s]['model_total'][j] for s in samples] for j in range(replicates)]
                maes = [mean_absolute_error(human_total_fluxes, model_total_fluxes[j]) for j in range(replicates)]
                plot_mae(i, k, domain_axs, maes)
                total_maes.append(maes)
            anova_tukey(total_maes, models)
            i += 1
            for c in classes:
                print(f'---{c}---')
                class_maes = []
                for k, m in enumerate(models):
                    human_class_fluxes = [flux_dict[d][m][s][c]['human'] for s in samples]
                    model_class_fluxes = [[flux_dict[d][m][s][c]['model'][j] for s in samples] for j in range(replicates)]
                    maes = [mean_absolute_error(human_class_fluxes, model_class_fluxes[j]) for j in range(replicates)]
                    plot_mae(i, k, domain_axs, maes)
                    class_maes.append(maes)
                anova_tukey(class_maes, models)
                i += 1

        domain_axs[0].axhline(human_measured_mae[d], color=black, alpha=0.3)
    
    legend_text = ('OOD', '+top1k', '+verify', '+minboost')
    lines = [Line2D([0], [0], color=colors[z], lw=6) for z, _ in enumerate(legend_text)]
    mae_fig.legend(lines, legend_text, ncol=len(legend_text), loc='outside upper center', frameon=False, handlelength=1)
    mae_fig.supylabel('MAE (mmol m$^{-2}$ d$^{-1}$)')   

    mae_fig.savefig(f'../results/figs/flux_comparison_mae.{image_format}', bbox_inches='tight')
    plt.close(mae_fig)


def relabeling_results():
    
    def compare_cols(df, col1, col2):
        
        n_matches = len(df[df[col1] == df[col2]])
        accuracy = n_matches / len(df) * 100
        
        return accuracy
            
    df = tools.load_metadata()

    for domain in ('RR', 'JC'):
        relabeled = df.loc[(df['domain'] == domain) & (df['relabel_group'].notnull())]
        r = compare_cols(relabeled, 'olabel_group', 'relabel_group')
        print(f'Intra-annotator agreement, {domain}: {r:.2f}')
    

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
        
    plt.savefig('../results/figs/map.pdf', bbox_inches='tight')
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

    open(f'../results/textfiles/metrics_hptune.txt', 'w')  # delete metrics file if it exists

    exp_dict = {'preprocessing': ['targetRR_ood', 'pad', 'normdata', 'normIN', 'padnormdata', 'padnormIN'],
                'learningrate': ['targetRR_ood', 'lowLR', 'highLR'],
                'weightdecay': ['targetRR_ood', 'lowWD', 'highWD']}
    
    offset = {'preprocessing': (-0.3, -0.18, -0.06, 0.06, 0.18, 0.3),
              'learningrate': (-0.2, 0, 0.2),
              'weightdecay': (-0.2, 0, 0.2)}

    labels = sorted(yaml.safe_load(open('../configs/targetRR_ood.yaml', 'r'))['classes'])
    y_vars = ('precision', 'recall')
    x_vars = labels + ['macro avg', 'weighted avg']
    colors = [blue, green, orange, vermillion, black, radish, sky]
    markers = ['o', '^', '+', 's', 'd', 'x', '*']

    for exp in exp_dict:

        cfg_names = exp_dict[exp]

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
            axs[i].set_ylabel(y.capitalize())
            if i != len(axs) - 1:
                axs[i].set_xticklabels([])
                axs[i].set_xticks(range(len(x_vars)))
            for j, x in enumerate(x_vars):
                with open(f'../results/textfiles/metrics_hptune.txt', 'a') as sys.stdout:
                    print(f'----{y}, {x}----')
                    for m, c in enumerate(cfg_names):
                        keys = [k for k in reports.keys() if f'{c}-' in k]
                        y_avg = np.mean([reports[k][x][y] for k in keys])
                        y_std = np.std([reports[k][x][y] for k in keys], ddof=1)
                        axs[i].errorbar(j + offset[exp][m], y_avg, y_std, color=colors[m], ecolor=colors[m], marker=markers[m], capsize=2)
                        print(f'{c}: {y_avg*100:.2f} ± {y_std*100:.2f}')
        
        lines = [Line2D([0], [0], color=colors[m], lw=6) for m, _ in enumerate(cfg_names)]
        leg_text = ['base' if x == 'targetRR_ood' else x for x in cfg_names]
        axs[0].legend(lines, leg_text, ncol=len(cfg_names), bbox_to_anchor=(0.5, 1.02), loc='lower center',
                      frameon=False, handlelength=1)      

        fig.savefig(f'../results/figs/metrics_{exp}.{image_format}')
        plt.close(fig)


def metrics_hitloop():

    open(f'../results/textfiles/metrics_hitloop.txt', 'w')  # delete metrics file if it exists

    offset = (-0.15, -0.05, 0.05, 0.15)
    replicates = 5
    y_vars = ('precision', 'recall')
    colors = [black, blue, green, orange, vermillion, radish, sky]
    markers = ['o', '^', '+', 's', 'd', 'x', '*']

    domains = ('RR', 'JC')
    labels = ('aggregate', 'fiber', 'long_pellet', 'mini_pellet', 'phytoplankton', 'rhizaria', 'salp_pellet', 'short_pellet', 'swimmer', 'unidentifiable')
    fig, axs = plt.subplots(len(y_vars) * len(domains), 1, layout='constrained', figsize=(10,10))
    axs_dict = {'RR': axs[:2], 'JC': axs[2:]}
    reports = {}

    for domain in domains:

        reports[domain] = {}
        d_axs = axs_dict[domain]

        flux_df = pd.read_csv(f'../results/fluxes/{domain}.csv', index_col='filename', low_memory=False)
        human_df = tools.load_metadata()
        relabel_df = human_df.loc[(human_df['domain'] == domain) & (human_df['relabel_group'].notnull())]
        relabel_report = classification_report(relabel_df['olabel_group'], relabel_df['relabel_group'], output_dict=True, zero_division=0)
        human_df = human_df.loc[human_df['domain'] == domain][['filename', 'olabel']]
        human_df = human_df.rename(columns={'olabel': f'label'})
        human_df.set_index('filename', inplace=True)
        models = (f'target{domain}_ood', f'target{domain}_top1k', f'target{domain}_verify', f'target{domain}_minboost')
        prediction_files = [f'{m}-{i}' for (m, i) in product(models, range(replicates))]

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
            cm.savefig(f'../results/figs/cmatrix_{f}.{image_format}')
            plt.close(cm)
            reports[domain][f] = classification_report(
                df['label'], df['prediction'], output_dict=True, zero_division=0, labels=labels)
        
        for i, y in enumerate(y_vars):
            d_axs[i].set_ylabel(f'{y.capitalize()} ({domain})')
            if i != len(axs) - 1:
                d_axs[i].set_xticklabels([])
                d_axs[i].set_xticks(range(len(labels)))
            for j, x in enumerate(labels):
                with open(f'../results/textfiles/metrics_hitloop.txt', 'a') as sys.stdout:
                    print(f'----{domain}, {y}, {x}----')
                    replicate_values_all_models = []
                    for z, m in enumerate(models):
                        keys = [k for k in reports[domain].keys() if f'{m}-' in k]
                        replicate_values = [reports[domain][k][x][y] for k in keys]
                        replicate_values_all_models.append(replicate_values)
                        y_avg = np.mean(replicate_values)
                        y_std = np.std(replicate_values, ddof=1)
                        d_axs[i].errorbar(j + offset[z], y_avg, y_std, color=colors[z], ecolor=colors[z], marker=markers[z], capsize=2)
                        print(f'{m}: {y_avg*100:.2f} ± {y_std*100:.2f}')
                    if x in relabel_df['olabel_group'].unique():  # JC didn't have any originally labeled salp pellets or rhizaria
                        d_axs[i].hlines(relabel_report[x][y], j + min(offset), j + max(offset), color=black, alpha=0.3)
                        print(f'relabel: {relabel_report[x][y]:.2f}')
                    anova_tukey(replicate_values_all_models, models)


    legend_text = ('OOD', '+top1k', '+verify', '+minboost')
    lines = [Line2D([0], [0], color=colors[z], lw=6) for z, _ in enumerate(models)]
    axs[-1].set_xticks(range(len(labels)), labels=labels, rotation=45)
    axs[0].legend(lines, legend_text, ncol=len(models), bbox_to_anchor=(0.5, 1.02), loc='lower center',
            frameon=False, handlelength=1)     

    fig.savefig(f'../results/figs/metrics_hitloop.{image_format}')
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


def flux_comparison_human_measured():
    
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    
    fig.supxlabel('Measured flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    fig.supylabel('Human flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)

    maes = {}

    for i, domain in enumerate(('RR', 'JC')):

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
            axs[i].errorbar(meas_flux, annot_flux, xerr=meas_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
            
        add_identity(axs[i], color=black, ls='--')
        mae = mean_absolute_error(meas_flux_allsamples, annot_flux_allsamples)
        maes[domain] = mae
        axs[i].text(0.98, 0.02, f'MAE: {mae:.2f}', ha='right', va='bottom', size=10, transform=transforms.blended_transform_factory(axs[i].transAxes, axs[i].transAxes))
        axs[i].text(0.02, 0.98, domain, ha='left', va='top', size=10, transform=transforms.blended_transform_factory(axs[i].transAxes, axs[i].transAxes))

    fig.savefig(f'../results/figs/flux_comparison_human.{image_format}', bbox_inches='tight')

    return maes


def flux_profiles():

    def mod_depths(real_depth):

        idx = (np.abs(acceptable_depths - real_depth)).argmin()
        
        return acceptable_depths[idx]
    
    
    def class_average(depth_df, col_prefix, averages, particle_class):

        df = depth_df.loc[depth_df[f'{col_prefix}_group'] == particle_class]
        samples = df['sample'].unique()
        if len(samples) > 0:
            sample_fluxes = []
            for s in samples:
                sample_fluxes.append(df.loc[df['sample'] == s][f'{col_prefix}_flux'].sum())
            averages.append(np.mean(sample_fluxes))
        else:
            averages.append(0)


    def measured_average(depth_df, averages, errors):

        samples = depth_df['sample'].unique()
        sample_fluxes = []
        sample_flux_errors = []
        for s in samples:
            sample_fluxes.append(depth_df.loc[depth_df['sample'] == s][f'measured_flux'].unique()[0])
            sample_flux_errors.append(
                depth_df.loc[depth_df['sample'] == s][f'measured_flux_e'].unique()[0])
        averages.append(np.mean(sample_fluxes))
        errors.append(np.sqrt(np.sum([err**2 for err in sample_flux_errors]))/len(sample_fluxes))


    ax_to_date = {0: [20180815], 1: [20180824], 2: [20180831],
                  3: [20210506, 20210508], 4: [20210514]}
    epoch_labels = ['RR, Epoch 1', 'RR, Epoch 2', 'RR, Epoch 3', 'JC, Epoch 1', 'JC, Epoch 2']
    jc_stts = ('JC5', 'JC6', 'JC7', 'JC8', 'JC21', 'JC22', 'JC23', 'JC24', 'JC25')
    xlims = [6, 6, 6, 15, 15]
    classes = ['aggregate', 'long_pellet', 'mini_pellet', 'phytoplankton', 'rhizaria', 'salp_pellet', 'short_pellet']
    colors = [blue, orange, vermillion, green, radish, sky, grey, black]
    acceptable_depths = np.array([75, 100, 125, 150, 175, 200, 330, 500])
    fig, axs_groups = plt.subplots(2, 5, layout='constrained')
    fig.supylabel('Depth (m)')
    fig.supxlabel('Flux (mmol m$^{-2}$ d$^{-1}$)')

    for i, axs in enumerate(axs_groups):

        for j, cl in enumerate(epoch_labels):
            axs[j].set_ylim(65, 510)
            axs[j].set_xlim(0, xlims[j])
            axs[j].invert_yaxis()
            domain = cl[:2]
            domain_df = pd.read_csv(f'../results/fluxes/{domain}.csv', index_col=False, low_memory=False)
            if domain == 'JC':  # only use STTs for JC
                domain_df = domain_df.loc[domain_df['sample'].isin(jc_stts)]
            if i == 0:
                axs[j].tick_params(labelbottom=False)
                if j == 0:
                    axs[j].set_ylabel('Model')
            else:
                axs[j].set_xlabel(cl)
                if j == 0:
                    axs[j].set_ylabel('Human')
            if j:
                axs[j].tick_params(labelleft=False)
            epoch_df = domain_df.loc[domain_df['date'].isin(ax_to_date[j])].copy()
            epoch_df['mod_depth'] = epoch_df.apply(lambda x: mod_depths(x['depth']), axis=1)
            depths = sorted(epoch_df['mod_depth'].unique())

            fill_from = np.zeros(len(depths))
            for k, c in enumerate(classes):
                class_flux_prof = []
                meas_flux_prof = []
                meas_flux_prof_e = []
                for d in depths:
                    depth_df = epoch_df.loc[epoch_df['mod_depth'] == d]
                    if i > 0:  # from human labels
                        col_prefix = 'olabel'
                        class_average(depth_df, col_prefix, class_flux_prof, c)
                        if k == 0:  # only need to calculate the measured fluxes once, c arg is irrelevant
                            measured_average(depth_df, meas_flux_prof, meas_flux_prof_e)
                    else:  # from model predictions
                        replicate_fluxes = []
                        for r in range(5):
                            col_prefix = f'predictiontarget{domain}_minboost-{r}'
                            class_average(depth_df, col_prefix, replicate_fluxes, c)
                        class_flux_prof.append(np.mean(replicate_fluxes))
                        if k == 0:
                            measured_average(depth_df, meas_flux_prof, meas_flux_prof_e)                                         
                fill_to = fill_from + class_flux_prof
                axs[j].fill_betweenx(depths, fill_from, fill_to, color=colors[k])
                fill_from = fill_to
                if k == 0:
                    axs[j].errorbar(meas_flux_prof, depths, xerr=meas_flux_prof_e, c=black, zorder=10)
    
    lines = [Line2D([0], [0], color=c, lw=6) for c in colors]
    # leg_text = ('aggregate', 'long_pellet', 'salp_pellet', 'other')
    leg_text = classes + ['measured']
    fig.legend(lines, leg_text, ncol=4, loc='outside upper center', frameon=False, handlelength=1)
    
    fig.savefig(f'../results/figs/flux_profiles.{image_format}', bbox_inches='tight')


if __name__ == '__main__':

    black = '#000000'
    orange = '#E69F00'
    sky = '#56B4E9'
    green = '#009E73'
    blue = '#0072B2'
    vermillion = '#D55E00'
    radish = '#CC79A7'
    white = '#FFFFFF'
    grey = '#A9A9A9'

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_format', '-i', default='png')
    args = parser.parse_args()
    image_format = args.image_format

    distribution_barplot()
    # draw_map()
    
    # training_plots()
    # metrics_hptune()
    # relabeling_results()
    
    # calculate_flux_df('RR')
    # calculate_flux_df('JC')

    # flux_comparison_human_measured()
    # flux_comparison_by_class()
    # metrics_hitloop()
    # flux_profiles()

