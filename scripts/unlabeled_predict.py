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
domain = 'RR'

metadata_df = pd.read_csv(os.path.join(cfg['data_dir'], 'metadata.csv'))
predict_df = metadata_df.loc[metadata_df['domain']=='RR'].copy()
predict_df['filepath'] = predict_df['label'] + '/' + predict_df['filename']
predict_fps = predict_df['filepath'].values
predict_dl = dataset.get_dataloader(cfg, predict_fps, is_labeled=False)

df_list = []

replicate = 0
saved_model_output = torch.load(m, map_location=device)
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

df = pd.DataFrame({'filename': y_fnames, f'prediction{replicate}': y_pred})
df.set_index('filename', inplace=True)
df['predictionR'] = np.random.choice(cfg['classes'], size=len(df))
df.to_csv('../results/predictions.csv')

