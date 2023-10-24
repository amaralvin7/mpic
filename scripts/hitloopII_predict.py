import os
import yaml

import src.dataset as dataset
import src.predict as predict

exp_name = 'hitloopII'
target_domain = 'RR'
cfg_fnames = sorted([c for c in os.listdir(f'../configs/{exp_name}') if '.yaml' in c])

for c in cfg_fnames:
    cfg = yaml.safe_load(open(f'../configs/{exp_name}/{c}', 'r'))
    models = [m for m in os.listdir(f'../results/{exp_name}/weights/') if c.split('.')[0] in m]
    for m in models:
        model_id = m.split('.')[0]
        fps = dataset.compile_filepaths(cfg, (target_domain,), 'test')
        loader = dataset.get_dataloader(cfg, fps, train=False, shuffle=False)
        predict.predict_labels(cfg['device'], loader, exp_name, model_id)
    