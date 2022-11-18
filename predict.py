
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import dataset
from model import initialize_model


def predict_with_truth(dataloader):
    
    y_pred = []
    y_true = []    

    with torch.no_grad():

        wrong_counter = 0
        
        for inputs, _, labels in tqdm(dataloader):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())
            
            # for i in range(len(preds)):
            #     if preds[i] != labels[i]:
            #         inp = inputs[i].numpy().transpose((1, 2, 0))
            #         inp = std * inp + mean
            #         inp = np.clip(inp, 0, 1)
            #         plt.imshow(inp)
            #         plt.suptitle(train_data.classes[preds[i]])
            #         preprocess.make_dir(f'wrong_predictions_true_labels/{train_data.classes[labels[i]]}')
            #         plt.savefig(f'wrong_predictions_true_labels/{train_data.classes[labels[i]]}/{wrong_counter}')
            #         plt.close()
            #         wrong_counter += 1

    report = classification_report(y_true, y_pred, zero_division=0, target_names=dataloader.dataset.classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv('test_output.csv')

    _, ax = plt.subplots(figsize=(10,10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=dataloader.dataset.classes,
        cmap=plt.cm.Blues,
        normalize=None,
        xticks_rotation='vertical',
        values_format='.0f',
        ax=ax
    )
    ax.set_title('Confusion matrix')
    plt.tight_layout()
    plt.savefig('confusionmatrix')
    plt.close()

    return y_pred, y_true


# def predict_without_truth(path):
    
#     test_loader = get_test_loader(UnlabeledImageFolder, path)
    
#     all_labels = []
#     all_filenames = []

#     with torch.no_grad():
        
#         for filenames, inputs in tqdm(test_loader):
#             outputs = model(inputs)
#             _, labels = torch.max(outputs, 1)
#             all_labels.extend([train_data.classes[i] for i in labels.tolist()])
#             all_filenames.extend(filenames)
    
#     df = pd.DataFrame(list(zip(all_filenames, all_labels)), columns =['file', 'predicted_label'])
    df.to_csv('predictions.csv', index=False)


def barplots(y_true, y_pred, dataloader):  # for precision, recall, f1 score

    def make_plot(metric_vals, metric_name):
        x = np.arange(len(classes))  # the label locations
        
        fig, ax = plt.subplots()
        ax.bar(x, metric_vals)
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.grid(axis='y', zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=90)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        fig.savefig(metric_name)
        plt.close()

    classes = dataloader.dataset.classes
    report = classification_report(y_true, y_pred, zero_division=0, target_names=classes, output_dict=True)
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    make_plot(precision, 'precision')
    make_plot(recall, 'recall')
    make_plot(f1, 'f1')

if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = yaml.safe_load(open('config.yaml', 'r'))
    
    trained_model_output = torch.load('saved_model.pt', map_location=device)
    weights = trained_model_output['weights']
    mean = trained_model_output['mean']
    std = trained_model_output['std']
    
    print(trained_model_output['train_loss_hist'])
    print('-------------')
    print(trained_model_output['train_acc_hist'])
    print('-------------')
    print(trained_model_output['val_loss_hist'])
    print('-------------')
    print(trained_model_output['val_acc_hist'])
    print('-------------')
    print(f'{len(trained_model_output["val_loss_hist"])} epochs, Acc: {100*max(trained_model_output["val_acc_hist"]):.2f}')

    test_filepaths =  dataset.stratified_split(cfg, 'test')
    test_dl = dataset.get_dataloader(cfg, test_filepaths, mean, std)
    
    model = initialize_model(len(test_dl.dataset.classes), device, weights=weights)
    model.eval()
    
    y_pred, y_true = predict_with_truth(test_dl)
    