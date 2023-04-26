import argparse
import os
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm

import src.dataset as dataset
from src.model import initialize_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='config.yaml')
args = parser.parse_args()
cfg = yaml.safe_load(open(os.path.join('..', args.config), 'r'))

mean, std = dataset.get_data_stats()
models = [f for f in os.listdir(os.path.join('..', 'weights')) if 'FULL' in f]

metadata_df = pd.read_csv(os.path.join(cfg['data_dir'], 'metadata.csv'))
predict_df = metadata_df.loc[metadata_df['subdir'] == 'none'].copy()
predict_df['filepath'] = predict_df['subdir'] + '/' + predict_df['filename']
predict_fps = predict_df['filepath'].values
predict_dl = dataset.get_dataloader(cfg, predict_fps, cfg['all_classes'], mean, std, is_labeled=False)

df_list = []

for m in models:

    replicate = m.split('.')[0][-1]
    saved_model_output = torch.load(
        os.path.join('..', 'weights', m), map_location=device)
    weights = saved_model_output['weights']
    model = initialize_model(len(cfg['all_classes']), weights=weights)
    model.eval()
            
    y_pred = []
    y_fnames = []

    print(f'Predicting experiment replicate {replicate}...')

    with torch.no_grad():

        for inputs, filepaths in tqdm(predict_dl):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend([predict_dl.dataset.idx_to_class[p] for p in preds.tolist()])
            y_fnames.extend([os.path.basename(f) for f in filepaths])

    df_list.append(pd.DataFrame({'filename': y_fnames, f'prediction_{replicate}': y_pred}))

dfs = [df.set_index('filename') for df in df_list]
merged = pd.concat(dfs, axis=1)

merged.to_csv('../results/unlabeled_predictions.csv')

