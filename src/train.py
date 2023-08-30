import argparse
import copy
import os

import numpy as np
import torch
import yaml

import src.dataset as dataset
import src.tools as tools
from src.model import initialize_model


def training_epoch(device, dataloader, model, optimizer, criterion, update):

    torch.set_grad_enabled(update)
    model.to(device)

    if update:
        model.train()
    else:
        model.eval()

    # running averages
    loss_total = 0.0
    acc_total = 0.0

    for inputs, _, labels in dataloader:

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if update:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # the .item() command retrieves the value of a single-valued tensor,
        # regardless of its data type and device of tensor
        loss_total += loss.item()
        # the predicted label is the one at position (class index) with highest
        # predicted value
        predictions = torch.argmax(outputs, dim=1)
        # number of correct predictions divided by batch size (i.e.,
        # average/mean)
        acc_total += torch.mean((predictions == labels).float()).item()

    loss_total /= len(dataloader)
    acc_total /= len(dataloader)

    return loss_total, acc_total


def train_model(cfg, device, model_arc, train_filepaths, val_filepaths, mean, std, exp_id, pad):

    classes = set([f.split('/')[0] for f in train_filepaths])
    train_dl = dataset.get_dataloader(cfg, train_filepaths, mean, std, augment=True, pad=pad)
    val_dl = dataset.get_dataloader(cfg, val_filepaths, mean, std, augment=False, pad=pad)
    model = initialize_model(model_arc, len(classes))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    tl_hist = []
    ta_hist = []
    vl_hist = []
    va_hist = []

    epoch = 1
    best_acc = 0
    best_loss = 10**15  # an arbitrarily large number
    esi = 0  # epochs since improvement

    train_start = tools.time_sync()

    while epoch <= cfg['max_epochs'] and esi < cfg['patience']:

        train_loss, train_acc = training_epoch(
            device, train_dl, model, optimizer, criterion, True)
        val_loss, val_acc = training_epoch(
            device, val_dl, model, optimizer, criterion, False)

        tl_hist.append(train_loss)
        ta_hist.append(train_acc)
        vl_hist.append(val_loss)
        va_hist.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
        if val_loss < best_loss:
            best_loss = val_loss
            esi = 0
        else:
            esi += 1

        epoch += 1

    train_duration = tools.time_sync() - train_start
    minutes = train_duration // 60
    seconds = train_duration % 60
    print(f'{exp_id}: {epoch - 1} epochs in {minutes:.0f}m {seconds:.0f}s. Best val acc {100*best_acc:.2f}%')

    output = {'mean': mean,
              'std': std,
              'train_loss_hist': tl_hist,
              'train_acc_hist': ta_hist,
              'val_loss_hist': vl_hist,
              'val_acc_hist': va_hist,
              'weights': best_weights}

    return output


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(os.path.join('..', args.config), 'r'))
    tools.set_seed(cfg, device)
    # experiment = Experiment(api_key=comet_key)
    train_fps, val_fps = dataset.compile_trainval_filepaths(cfg, cfg['train_domains'])
    mean = cfg['mean']
    std = cfg['std']
    model_arc = cfg['model']
    batch_size = cfg['batch_size']
    pad = True
    if mean == 'None' and std == 'None':
        mean = None
        std = None

    exp_id = args.config.split('.')[0].split('_')[1]
    print(f'---------Training Models (exp_id={exp_id})...')
    output = train_model(cfg, device, model_arc, train_fps, val_fps, mean, std, exp_id, pad)
    torch.save(output, os.path.join('..', 'results', 'weights', f'savedmodel_{exp_id}.pt'))