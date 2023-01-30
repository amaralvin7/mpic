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

train_split = 'RR_SRT_FK_JC'  # RR, SRT, FK train split
mean, std = dataset.get_train_data_stats(cfg, train_split)
models = [f for f in os.listdir(os.path.join('..', 'weights')) if f'model_{train_split}' in f]

unlabeled_data_dir = '../jc_unlabeled'
predict_fps = [f for f in os.listdir(unlabeled_data_dir) if f.split('.')[1] in cfg['exts']]
predict_dl = dataset.get_dataloader(cfg, unlabeled_data_dir, predict_fps, mean, std, augment=False, is_labeled=False)

df_list = []

for m in models:

    replicate = m.split('.')[0][-1]
    saved_model_output = torch.load(
        os.path.join('..', 'weights', m), map_location=device)
    weights = saved_model_output['weights']
    model = initialize_model(len(cfg['classes']), weights=weights)
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

    df_list.append(pd.DataFrame({'file': y_fnames, f'prediction_{replicate}': y_pred}))

dfs = [df.set_index('file') for df in df_list]
merged = pd.concat(dfs, axis=1)

merged.to_csv('JC_predictions.csv')

