import argparse
import os
import yaml

import src.dataset as dataset
import src.predict as predict
import src.tools as tools

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_list_id', '-i')
args = parser.parse_args()
list_id = int(args.model_name_list_id)

model_names = tools.get_model_names(list_id)

for model_name in model_names:
    cfg = yaml.safe_load(open(f'../configs/{model_name}.yaml', 'r'))
    models = [m for m in os.listdir(f'../results/weights/') if model_name in m]
    for m in models:
        model_id = m.split('.')[0]
        fps = dataset.compile_filepaths(cfg, 'test')
        loader = dataset.get_dataloader(cfg, fps, train=False, shuffle=False)
        predict.predict_labels(cfg['device'], loader, model_id)
    