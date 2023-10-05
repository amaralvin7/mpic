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
from sklearn.metrics import ConfusionMatrixDisplay, mean_squared_error, classification_report
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

import src.dataset as dataset
import src.predict as predict
import src.tools as tools


def training_plots():

    models = [f for f in os.listdir(os.path.join('..', 'results', 'weights')) if f'.pt' in f]

    for m in models:
        exp_id = m.split('.')[0]
        model_output = torch.load(os.path.join('..', 'results', 'weights', m),
                                  map_location='cpu')
        num_epochs = len(model_output['train_loss_hist'])

        plt.xlabel('Training Epochs')
        plt.ylabel('Accuracy')
        plt.plot(range(1, num_epochs + 1), model_output['train_acc_hist'],
                 label='training')
        plt.plot(range(1, num_epochs + 1), model_output['val_acc_hist'],
                 label='validation')
        plt.legend()
        plt.savefig(os.path.join('..', 'results', 'figs', f'accuracy_{exp_id}.png'))
        plt.close()

        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.plot(range(1, num_epochs + 1), model_output['train_loss_hist'],
                 label='training')
        plt.plot(range(1, num_epochs + 1), model_output['val_loss_hist'],
                 label='validation')
        plt.legend()
        plt.savefig(os.path.join('..', 'results', 'figs', f'loss_{exp_id}.png'))
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
        print(row['filename'])
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


def calculate_flux_df(cfg, domain=None):
    
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
    if domain is not None:
        metadata = metadata.loc[metadata['domain']==domain]
    predictions = pd.read_csv('../results/predictions.csv')
    df = metadata.merge(predictions, how='left', on='filename')
    df = df.loc[df['esd'].notnull()].copy()  # 23 filenames are in the data folder but not in the metadata
    pred_columns = [c for c in df.columns if 'pred' in c]
    tqdm.pandas()
    df = df.progress_apply(row_flux, axis=1)
    # df = df.apply(row_flux, axis=1)
    
    df.to_csv('../results/fluxes.csv', index=False)


def flux_comparison():
    
    fig, axs = plt.subplots(1, 3, figsize=(12,4))
    fig.subplots_adjust(left=0.1, wspace=0.2)
    
    fig.supylabel('Predicted flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    axs[0].set_xlabel('Measured flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    axs[1].set_xlabel('Annotated flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    axs[2].set_xlabel('Random flux (mmol m$^{-2}$ d$^{-1}$)', fontsize=14)
    
    df = pd.read_csv('../results/fluxes.csv', index_col=False, low_memory=False)
            
    for s in df['sample'].unique():
        
        # if s[0] == 'J' and int(s[2:]) >= 49:
        #     continue

        sdf = df.loc[(df['sample'] == s)].copy()
        meas_flux = sdf['measured_flux'].unique()[0]
        meas_flux_e = sdf['measured_flux_e'].unique()[0]
        annot_flux = sdf['olabel_flux'].sum()
        pred_flux = sdf['prediction0_flux'].sum()
        rand_flux = sdf['predictionR_flux'].sum()
        
        color = get_domain_color(sdf['domain'].unique()[0])
        axs[0].errorbar(meas_flux, pred_flux, xerr=meas_flux_e, c=color, fmt='o', elinewidth=1, ms=4, capsize=2)
        axs[1].scatter(annot_flux, pred_flux, c=color, s=4)
        axs[2].scatter(rand_flux, pred_flux, c=color, s=4)
        


    for ax in axs:
        add_identity(ax, color=black, ls='--')
        # ax.set_yscale('log')
        # ax.set_xscale('log')
            

    # lines = [Line2D([0], [0], color=orange, lw=4),
    #          Line2D([0], [0], color=vermillion, lw=4),
    #          Line2D([0], [0], color=blue, lw=4),
    #          Line2D([0], [0], color=green, lw=4)]
    # labels = ['FK', 'JC', 'RR', 'SR']
    # axs[0].legend(lines, labels, frameon=False, handlelength=1)

    fig.savefig(f'../results/figs/flux_comparison.png', bbox_inches='tight')

    # print(np.sqrt(mean_squared_error(rrfk_orig, rrfk_pred)))
    # print(np.sqrt(mean_squared_error(all_meas, all_pred)))


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


def compare_accuracies():
    
    def compare_cols(df, col1, col2):
        
        n_matches = len(df[df[col1] == df[col2]])
        accuracy = n_matches / len(df) * 100
        
        return accuracy
            
    df = pd.read_csv('../results/fluxes.csv', index_col=False)
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


def metrics_by_exp():

    for split in ('test', 'val'):
        
        df = pd.read_csv(f'../results/predictions_{split}.csv')
        df = df.loc[df['label'] != 'none']

        color_dict = {'base': green, 'normdata': orange, 'normimagenet': blue}
        y_vars = ('precision', 'recall', 'f1-score')
        x_vars = ['macro avg', 'weighted avg']
        
        _ , axs = plt.subplots(len(y_vars), len(x_vars))
        for k, exp in enumerate(color_dict.keys()):
            labels = yaml.safe_load(open(f'../configs/config_{exp}.yaml', 'r'))['classes']
            report = classification_report(df['label'], df[f'prediction_{exp}'], output_dict=True, zero_division=0, labels=labels)
            for j, x in enumerate(x_vars):
                axs[-1,j].set_xlabel(x)
                for i, y in enumerate(y_vars):
                    metric = report[x][y]
                    axs[i,j].bar(k, metric, width=1, color=color_dict[exp])
                    axs[i,j].set_ylim(0, 1.2)
                    axs[i,j].set_xticks([])
                    axs[i,j].text(k, metric, f'{metric:.2f}', ha='center', va='bottom', size=10)
                    if j == 0:
                        axs[i,j].set_ylabel(y)

        lines = [Line2D([0], [0], color=color_dict[c], lw=6) for c in color_dict.keys()]
        axs[0][1].legend(lines, color_dict.keys(), ncol=6, bbox_to_anchor=(0, 1.02), loc='lower center',
                frameon=False, handlelength=1)

        plt.savefig(f'../results/figs/metrics_{split}.png')
        plt.close()

        for exp in color_dict.keys():
            _, ax = plt.subplots(figsize=(8, 8))
            ConfusionMatrixDisplay.from_predictions(
                df['label'],
                df[f'prediction_{exp}'],
                cmap=plt.cm.Blues,
                normalize=None,
                xticks_rotation='vertical',
                values_format='.0f',
                ax=ax)
            ax.set_title('Perfect labels vs. predictions')
            plt.tight_layout()
            plt.savefig(f'../results/figs/confusionmatrix_{exp}_{split}.png')
            plt.close()


def metrics():

    exp_dict = {'preprocessing': ['base', 'pad', 'normdata', 'normIN', 'pad_normdata', 'pad_normIN'],
                'learningrate': ['base', 'highLR', 'lowLR'],
                'weightdecay': ['base', 'highWD', 'lowWD'],
                'upsample': ['base', 'upsample100', 'upsample200', 'upsample400']}

    labels = yaml.safe_load(open('../configs/base.yaml', 'r'))['classes']
    y_vars = ('precision', 'recall')
    x_vars = labels + ['macro avg', 'weighted avg']
    colors = [blue, green, orange, vermillion, black, radish, sky]
    markers = ['o', '^', '+', 's', 'd', 'x', '*']

    for exp, s in product(exp_dict, ('target', 'test')):

        cfg_names = sorted(exp_dict[exp])

        fig, axs = plt.subplots(len(y_vars), 1, tight_layout=True, figsize=(10,5))
        axs[-1].set_xticks(range(len(x_vars)), labels=x_vars, rotation=45)
        prediction_files = [f for f in os.listdir('../results/predictions') if s in f and f.split('-')[0].split(f'{s}_')[1] in cfg_names]
        reports = {}

        for f in prediction_files:
            df = pd.read_csv(f'../results/predictions/{f}')
            df['label'] = df['filepath'].apply(lambda x: x.split('/')[0])
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
                print(f'----{s}, {y}, {x}----')
                for m, c in enumerate(cfg_names):
                    keys = [k for k in reports.keys() if f'{s}_{c}-' in k]
                    y_avg = np.mean([reports[k][x][y] for k in keys])
                    y_std = np.std([reports[k][x][y] for k in keys], ddof=1)
                    axs[i].errorbar(j, y_avg, y_std, color=colors[m], ecolor=colors[m], marker=markers[m], capsize=2)
                    print(f'{c}: {y_avg*100:.2f} ± {y_std*100:.2f}')
        
        lines = [Line2D([0], [0], color=colors[m], lw=6) for m, _ in enumerate(cfg_names)]
        axs[0].legend(lines, cfg_names, ncol=len(cfg_names), bbox_to_anchor=(0.5, 1.02), loc='lower center',
                frameon=False, handlelength=1)      

        fig.savefig(f'../results/figs/metrics_{s}_{exp}.{out_form}')
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

if __name__ == '__main__':

    black = '#000000'
    orange = '#E69F00'
    sky = '#56B4E9'
    green = '#009E73'
    blue = '#0072B2'
    vermillion = '#D55E00'
    radish = '#CC79A7'
    white = '#FFFFFF'

    out_form = 'pdf'

    training_plots()
    metrics()
    # calculate_flux_df(cfg, domain='RR')
    # compare_accuracies()
    # softmax_histograms(cfg)
