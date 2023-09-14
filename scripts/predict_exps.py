import os

import pandas as pd
import yaml
from itertools import product

import src.dataset as dataset
import src.predict as predict

models = [m for m in os.listdir('../results/weights/') if '.pt' in m]

for split, m in product(('test', 'target'), models):

    exp_id = m.split('.')[0]
    cfg = yaml.safe_load(open(f'../configs/{exp_id.split("-")[0]}.yaml', 'r'))
    if split == 'test':
        fps = dataset.compile_filepaths(cfg, cfg['train_domains'], 'test')
    else:
        fps = []
        for s in ('train', 'val', 'test'):
            fps.extend(dataset.compile_filepaths(cfg, (cfg['target_domain'],), s))
    loader = dataset.get_dataloader(cfg, fps, shuffle=False)
    y_fp, y_scores, y_pred, _ = predict.predict_labeled_data(cfg['device'], loader, m)
    new_fp = []
    for f in y_fp:
        groups = f.split('/')
        new_fp.append(f'{groups[-2]}/{groups[-1]}')
    y_pred = [loader.dataset.idx_to_class[p] for p in y_pred]
    df = pd.DataFrame({'filepath': new_fp, 'prediction': y_pred})

    for i in range(y_scores.shape[1]):
        df[f'{loader.dataset.idx_to_class[i]}'] = y_scores[:,i]
        
    df.set_index('filepath', inplace=True)
    df.to_csv(f'../results/predictions/{split}_{exp_id}.csv')

    # merged = pd.concat(df_list, axis=1)
    # merged['filepath'] = merged.index.to_series()
    # merged['filename'] = merged['filepath'].apply(lambda x: os.path.basename(x))
    # merged['label'] = merged['filepath'].apply(lambda x: os.path.basename(os.path.split(x)[0]))
    # merged.set_index('filename', inplace=True)
    # merged.drop('filepath', axis=1, inplace=True)
    

