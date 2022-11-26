
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from itertools import product

import pandas as pd
import torch
import sklearn.metrics as metrics
import yaml
from PIL import Image
from tqdm import tqdm

import dataset
import tools
from model import initialize_model
from colors import *


def predict_with_truth(dataloader, model, exp_id, replicate):

    print(f'Predicting experiment {exp_id}, replicate {replicate}...')
    
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

    classes = dataloader.dataset.classes
    class_idxs = list(range(len(classes)))
    report = metrics.classification_report(y_true, y_pred, zero_division=0, labels=class_idxs, target_names=classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join('results', f'pred_output_{exp_id}_{replicate}.csv'))

    _, ax = plt.subplots(figsize=(10,10))
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=class_idxs,
        display_labels=classes,
        cmap=plt.cm.Blues,
        normalize=None,
        xticks_rotation='vertical',
        values_format='.0f',
        ax=ax
    )
    ax.set_title('Confusion matrix')
    plt.tight_layout()
    plt.savefig(os.path.join('results', f'confusionmatrix_{exp_id}_{replicate}'))
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
#     df.to_csv('predictions.csv', index=False)


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
    report = metrics.classification_report(y_true, y_pred, zero_division=0, target_names=classes, output_dict=True)
    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]
    make_plot(precision, 'precision')
    make_plot(recall, 'recall')
    make_plot(f1, 'f1')


def get_experiment_matrix(cfg):
    
    matrix ={}
    combos = product(cfg['train_splits'], cfg['train_domains'])
    for i, combo in enumerate(combos):
        matrix[i] = [*combo]
        
    return matrix


def prediction_experiments(cfg, device, filename):

    exp_matrix = get_experiment_matrix(cfg)
    
    test_acc_avgs = []
    macro_f1_avgs = []
    weight_f1_avgs = []

    test_acc_std = []
    macro_f1_std = []
    weight_f1_std = []
    
    for exp_id, (split_id, predict_domain) in exp_matrix.items():
        
        e_test_acc = []
        e_macro_f1 = []
        e_weight_f1 = []

        mean, std = dataset.get_train_data_stats(cfg, split_id)
        predict_fps =  dataset.get_predict_filepaths(cfg, predict_domain)
        predict_dl = dataset.get_dataloader(cfg, predict_fps, mean, std)
        models = [f for f in os.listdir('results') if f'model_{split_id}' in f]

        for m in sorted(models):
            
            replicate = m.split('.')[0][-1]
            model_output = torch.load(os.path.join('results', m), map_location=device)
            weights = model_output['weights']
            model = initialize_model(len(cfg['classes']), weights=weights)
            model.eval()
            
            y_pred, y_true = predict_with_truth(predict_dl, model, exp_id, replicate)
            e_test_acc.append(metrics.accuracy_score(y_true, y_pred))
            e_macro_f1.append(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0))
            e_weight_f1.append(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0))

        test_acc_avgs.append(np.mean(e_test_acc))
        macro_f1_avgs.append(np.mean(e_macro_f1))
        weight_f1_avgs.append(np.mean(e_weight_f1))

        test_acc_std.append(np.std(e_test_acc, ddof=1))
        macro_f1_std.append(np.std(e_macro_f1, ddof=1))
        weight_f1_std.append(np.std(e_weight_f1, ddof=1))
    
    prediction_results = {'taa': test_acc_avgs, 'mfa': macro_f1_avgs,
                          'wfa': weight_f1_avgs, 'tas': test_acc_std,
                          'mfs': macro_f1_std, 'wfs': weight_f1_std}

    tools.write_json(prediction_results, os.path.join('results', filename))


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    prediction_experiments(cfg, device, 'prediction_results.json')
    
    