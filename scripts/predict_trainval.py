import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import src.dataset as dataset
from src.model import initialize_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg = yaml.safe_load(open('../config.yaml', 'r'))
m = '../results/weights/savedmodel_padTrue-0.pt'

metadata_df = pd.read_csv(os.path.join(cfg['data_dir'], 'metadata.csv'))
train_fps, val_fps = dataset.compile_trainval_filepaths(cfg)
train_dl = dataset.get_dataloader(cfg, train_fps, is_labeled=False)
val_dl = dataset.get_dataloader(cfg, val_fps, is_labeled=False)

data = {'train': train_dl, 'val': val_dl}

saved_model_output = torch.load(m, map_location=device)
weights = saved_model_output['weights']
model = initialize_model(len(cfg['classes']), weights=weights)
model.eval()
        
y_pred = []
y_fnames = []
y_labels = []

for split in data:

    print(f'Predicting on {split} images...')

    dataloader = data[split]

    with torch.no_grad():

        for inputs, filepaths in tqdm(dataloader):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend([dataloader.dataset.idx_to_class[p] for p in preds.tolist()])
            y_fnames.extend([os.path.basename(f) for f in filepaths])
            y_labels.extend([os.path.basename(os.path.split(f)[0]) for f in filepaths])

    df = pd.DataFrame({'filename': y_fnames, 'label': y_labels, f'prediction_{split}': y_pred})
    df.to_csv(f'../results/predictions_{split}.csv')

