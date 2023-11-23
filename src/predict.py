import numpy as np
import pandas as pd
import torch

from src.model import initialize_model


def predict_labels(device, dataloader, model_id, prefix=None):

    csv_fname = f'{prefix}_{model_id}' if prefix is not None else model_id

    model_output = torch.load(f'../results/weights/{model_id}.pt', map_location=device)
    weights = model_output['weights']
    model = initialize_model(len(dataloader.dataset.classes), weights=weights)
    model.eval()

    print(f'Predicting with {csv_fname}...')

    y_pred = []
    y_fp = []
    y_scores = []

    with torch.no_grad():

        for inputs, filepaths in dataloader:
            outputs = torch.nn.Softmax(dim=1)(model(inputs))
            _, preds = torch.max(outputs, 1)
            y_scores.extend(outputs)
            y_pred.extend(preds.tolist())
            y_fp.extend(filepaths)
    
    y_scores = np.stack(y_scores, axis=0)
    y_pred = [dataloader.dataset.idx_to_class[p] for p in y_pred]

    df = pd.DataFrame({'filepath': y_fp, 'prediction': y_pred})

    for i in range(y_scores.shape[1]):
        df[f'{dataloader.dataset.idx_to_class[i]}'] = y_scores[:,i]
        
    df.set_index('filepath', inplace=True)
    df.to_csv(f'../results/predictions/{csv_fname}.csv')
    
