import os
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm

import src.dataset as dataset
import src.predict as predict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = [f for f in os.listdir(f'../results/weights') if 'pt' in f]

for split in ('test', 'val'):
    df_list = []
    for m in models:
        exp_id = m.split('.')[0].split('_')[1]
        cfg = yaml.safe_load(open(f'../config_{exp_id}.yaml', 'r'))
        if split == 'test':
            fps = dataset.compile_trainvaltest_filepaths(cfg, 'RR')
        else:
            _, fps = dataset.compile_trainval_filepaths(cfg, cfg['train_domains'])
        mean = cfg['mean']
        std = cfg['mean']
        model_arc = cfg['model']
        if mean == 'None' and std == 'None':
            mean = None
            std = None
        loader = dataset.get_dataloader(cfg, fps, mean, std, pad=True)
        y_fp, y_pred, _ = predict.predict_labeled_data(device, loader, m, model_arc)
        y_pred = [loader.dataset.idx_to_class[p] for p in y_pred]
        df = pd.DataFrame({'filepath': y_fp, f'prediction_{exp_id}': y_pred})
        df.set_index('filepath', inplace=True)
        df_list.append(df)

        merged = pd.concat(df_list, axis=1)
        merged['filepath'] = merged.index.to_series()
        merged['filename'] = merged['filepath'].apply(lambda x: os.path.basename(x))
        merged['label'] = merged['filepath'].apply(lambda x: os.path.basename(os.path.split(x)[0]))
        merged.set_index('filename', inplace=True)
        merged.drop('filepath', axis=1, inplace=True)
        merged.to_csv(f'../results/predictions_{split}.csv')

