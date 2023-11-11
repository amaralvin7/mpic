import os
import yaml

import src.dataset as dataset
import src.predict as predict
import src.tools as tools


exp_name = 'hitloopI'
target_domain = 'RR'

metadata = tools.load_metadata()

df = metadata.loc[metadata['domain'] == target_domain]  # target is all RR images
fps = df['label'] + '/' + df['filename']
fps = fps.values

models = [m for m in os.listdir(f'../results/{exp_name}/weights/') if '.pt' in m]

for m in models:

    model_id = m.split('.')[0]
    model_name = model_id.split('-')[0]

    cfg = yaml.safe_load(open(f'../configs/{exp_name}/{model_name}.yaml', 'r'))
    
    loader = dataset.get_dataloader(cfg, fps, train=False, shuffle=False)
    predict.predict_labels(cfg['device'], loader, exp_name, model_id)
    