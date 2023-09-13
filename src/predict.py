import numpy as np
import torch
from tqdm import tqdm

from src.model import initialize_model


def predict_labeled_data(device, dataloader, model_fn):

    model_output = torch.load(f'../results/weights/{model_fn}', map_location=device)
    weights = model_output['weights']
    model = initialize_model(len(dataloader.dataset.classes), weights=weights)
    model.eval()

    print(f'Predicting with {model_fn}...')

    y_pred = []
    y_true = []
    y_fp = []
    y_scores = []

    with torch.no_grad():

        for inputs, filepaths, labels in tqdm(dataloader):
            outputs = torch.nn.Softmax(dim=1)(model(inputs))
            _, preds = torch.max(outputs, 1)
            y_scores.extend(outputs)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
            y_fp.extend(filepaths)
    
    y_scores = np.stack(y_scores, axis=0)

    return y_fp, y_scores, y_pred, y_true
    
