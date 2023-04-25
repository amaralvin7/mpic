import matplotlib.pyplot as plt
import numpy as np
import os
from itertools import product

import pandas as pd
import torch
import sklearn.metrics as metrics

import src.dataset as dataset
import src.tools as tools
from src.model import initialize_model


def predict_labeled_data(dataloader, model, exp_id, replicate):

    print(f'Predicting experiment {exp_id}, replicate {replicate}...')

    y_pred = []
    y_true = []

    with torch.no_grad():

        for inputs, _, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    # classes = dataloader.dataset.classes
    # class_idxs = list(range(len(classes)))
    # report = metrics.classification_report(
    #     y_true,
    #     y_pred,
    #     zero_division=0,
    #     labels=class_idxs,
    #     target_names=classes,
    #     output_dict=True)

    # pd.DataFrame(report).transpose().to_csv(
    #     os.path.join('..', 'results', f'pred_output_{exp_id}_{replicate}.csv'))

    # _, ax = plt.subplots(figsize=(10, 10))
    # metrics.ConfusionMatrixDisplay.from_predictions(
    #     y_true,
    #     y_pred,
    #     labels=class_idxs,
    #     display_labels=classes,
    #     cmap=plt.cm.Blues,
    #     normalize=None,
    #     xticks_rotation='vertical',
    #     values_format='.0f',
    #     ax=ax)
    # ax.set_title('Confusion matrix')
    # plt.tight_layout()
    # plt.savefig(
    #     os.path.join('..', 'results', f'confusionmatrix_{exp_id}_{replicate}'))
    # plt.close()

    return y_pred, y_true


def get_experiment_matrix(cfg):

    matrix = {}
    train_splits = dataset.powerset(cfg['domains'])
    train_split_ids = [('_').join(domains) for domains in train_splits]
    combos = product(train_split_ids, cfg['domains'])
    for i, combo in enumerate(combos):
        matrix[i] = [*combo]

    return matrix


def prediction_experiments(cfg, device, exp_matrix, save_fname, uniform=False):

    test_acc_avgs = []
    macro_f1_avgs = []
    weight_f1_avgs = []

    test_acc_std = []
    macro_f1_std = []
    weight_f1_std = []
    
    uni_suffix = 'u' if uniform else ''

    for exp_id, (split_id, predict_domain) in exp_matrix.items():

        e_test_acc = []
        e_macro_f1 = []
        e_weight_f1 = []

        mean, std = dataset.get_data_stats(split_id)
        predict_fps = dataset.get_predict_filepaths(cfg, predict_domain, cfg['ablation_classes'])
        predict_dl = dataset.get_dataloader(cfg, predict_fps, cfg['ablation_classes'], mean, std)
        models = [f for f in os.listdir(os.path.join('..', 'weights')) if f'model_{split_id}{uni_suffix}-' in f]

        for m in sorted(models):

            replicate = m.split('.')[0][-1]
            model_output = torch.load(
                os.path.join('..', 'weights', m), map_location=device)
            weights = model_output['weights']
            model = initialize_model(len(predict_dl.dataset.classes), weights=weights)
            model.eval()

            y_pred, y_true = predict_labeled_data(
                predict_dl, model, exp_id, replicate)
            e_test_acc.append(metrics.accuracy_score(y_true, y_pred))
            e_macro_f1.append(
                metrics.f1_score(
                    y_true,
                    y_pred,
                    average='macro',
                    zero_division=0))
            e_weight_f1.append(
                metrics.f1_score(
                    y_true,
                    y_pred,
                    average='weighted',
                    zero_division=0))

        test_acc_avgs.append(np.mean(e_test_acc))
        macro_f1_avgs.append(np.mean(e_macro_f1))
        weight_f1_avgs.append(np.mean(e_weight_f1))

        test_acc_std.append(np.std(e_test_acc, ddof=1))
        macro_f1_std.append(np.std(e_macro_f1, ddof=1))
        weight_f1_std.append(np.std(e_weight_f1, ddof=1))

    prediction_results = {'taa': test_acc_avgs, 'mfa': macro_f1_avgs,
                          'wfa': weight_f1_avgs, 'tas': test_acc_std,
                          'mfs': macro_f1_std, 'wfs': weight_f1_std}

    tools.write_json(
        prediction_results,
        os.path.join('..', 'results', save_fname))
    
