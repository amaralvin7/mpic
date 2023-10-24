import os

import pandas as pd
import yaml
from itertools import product

import src.dataset as dataset
import src.predict as predict

exp_name = 'hptune'
models = sorted([m for m in os.listdir(f'../results/{exp_name}/weights/') if '.pt' in m])
target_domain = 'RR'

for split, m in product(('test', 'target'), models):

    model_id = m.split('.')[0]
    cfg = yaml.safe_load(open(f'../configs/{exp_name}/{model_id.split("-")[0]}.yaml', 'r'))
    if split == 'test':
        fps = dataset.compile_filepaths(cfg, cfg['train_domains'], 'test')
    else:
        fps = []
        for s in ('train', 'val', 'test'):
            fps.extend(dataset.compile_filepaths(cfg, (target_domain,), s))
    loader = dataset.get_dataloader(cfg, fps, train=False, shuffle=False)
    predict.predict_labels(cfg['device'], loader, exp_name, model_id, prefix=split)
    