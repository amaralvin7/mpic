import os
import sys

import pandas as pd
import torch
import yaml
from tqdm import tqdm
import numpy as np

import src.dataset as dataset
import src.predict as predict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

models = ['savedmodel_base.pt', 'savedmodel_batchsize.pt', 'savedmodel_upsample.pt']

for split in ('test', 'val'):
    df_list = []
    for m in models:
        exp_id = m.split('.')[0].split('_')[1]
        cfg = yaml.safe_load(open(f'../configs/config_{exp_id}.yaml', 'r'))
        if split == 'test':
            fps = dataset.compile_trainvaltest_filepaths(cfg, 'RR')
        else:
            _, fps = dataset.compile_trainval_filepaths(cfg, cfg['train_domains'])
        mean = cfg['mean']
        std = cfg['std']
        model_arc = cfg['model']
        loader = dataset.get_dataloader(cfg, fps, mean, std, pad=True)
        y_fp, y_pred, _ = predict.predict_labeled_data(device, loader, m, model_arc)
        new_fp = []
        for f in y_fp:
            groups = f.split('/')
            new_fp.append(f'{groups[-2]}/{groups[-1]}')
        y_pred = [loader.dataset.idx_to_class[p] for p in y_pred]
        df = pd.DataFrame({'filepath': new_fp, f'prediction_{exp_id}': y_pred})
        df.set_index('filepath', inplace=True)
        df_list.append(df)

    merged = pd.concat(df_list, axis=1)
    merged['filepath'] = merged.index.to_series()
    merged['filename'] = merged['filepath'].apply(lambda x: os.path.basename(x))
    merged['label'] = merged['filepath'].apply(lambda x: os.path.basename(os.path.split(x)[0]))
    merged.set_index('filename', inplace=True)
    merged.drop('filepath', axis=1, inplace=True)
    merged.to_csv(f'../results/predictions_{split}.csv')

