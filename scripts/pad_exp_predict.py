import os
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm

import src.dataset as dataset
import src.predict as predict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = yaml.safe_load(open('../config.yaml', 'r'))
mean = None
std = None

predict_fps = dataset.compile_trainvaltest_filepaths(cfg, 'RR')
dl_padF = dataset.get_dataloader(cfg, predict_fps, mean, std, pad=False)
dl_padT = dataset.get_dataloader(cfg, predict_fps, mean, std, pad=True)

df_list = []
run_folder = 'pad_exp_targetRR'
models = [f for f in os.listdir(f'../runs/{run_folder}/weights') if 'pt' in f]

for m in models:
    _, model_id = m.split('_')
    padbool, replicate = model_id.split('-')
    if padbool == 'padTrue':
        loader = dataset.get_dataloader(cfg, predict_fps, mean, std, pad=True)
    else:
        loader = dataset.get_dataloader(cfg, predict_fps, mean, std, pad=False)
    y_fp, y_pred, _ = predict.predict_labeled_data(device, loader, run_folder, m)
    y_pred = [loader.dataset.idx_to_class[p] for p in y_pred]
    df = pd.DataFrame({'filepath': y_fp, f'prediction_{model_id}': y_pred})
    df.set_index('filepath', inplace=True)
    df_list.append(df)

merged = pd.concat(df_list, axis=1)
merged['filepath'] = merged.index.to_series()
merged['filename'] = merged['filepath'].apply(lambda x: os.path.basename(x))
merged['label'] = merged['filepath'].apply(lambda x: os.path.basename(os.path.split(x)[0]))
merged.set_index('filename', inplace=True)
merged.drop('filepath', axis=1, inplace=True)
merged.to_csv(f'../results/figs/pad_exp_predictions.csv')
