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
predict_df = metadata_df.loc[metadata_df['domain']==cfg['test_domain']].copy()
predict_df['filepath'] = predict_df['label'] + '/' + predict_df['filename']
predict_fps = predict_df['filepath'].values
predict_dl = dataset.get_dataloader(cfg, predict_fps, is_labeled=False)

replicate = 0
saved_model_output = torch.load(m, map_location=device)
weights = saved_model_output['weights']
model = initialize_model(len(cfg['classes']), weights=weights)
model.eval()
        
y_fnames = []
y_fps = []
y_scores = []
y_preds = []

print(f'Predicting experiment replicate {replicate}...')

with torch.no_grad():

    for inputs, filepaths in tqdm(predict_dl):
        outputs = torch.nn.Softmax(dim=1)(model(inputs))
        _, preds = torch.max(outputs, 1)
        y_scores.extend(outputs)
        y_preds.extend([predict_dl.dataset.idx_to_class[p] for p in preds.tolist()])
        y_fnames.extend([os.path.basename(f) for f in filepaths])
        y_fps.extend(filepaths)

y_scores = np.stack(y_scores, axis=0)
df = pd.DataFrame({'filename': y_fnames, 'filepath': y_fps, 'prediction0': y_preds})
for i in range(y_scores.shape[1]):
    df[f'{predict_dl.dataset.idx_to_class[i]}'] = y_scores[:,i]
df.set_index('filename', inplace=True)
df['predictionR'] = np.random.choice(cfg['classes'], size=len(df))
df.to_csv('../results/predictions_test.csv')

