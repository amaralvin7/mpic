import argparse
import copy
import csv
import os

import torch
import yaml

import dataset
import tools
from model import initialize_model


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

    for _, (inputs, _, labels) in enumerate(dataloader):

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


def write_filenames_to_csv(filepaths):

    filenames = [os.path.basename(f) for f in filepaths]
    with open('filenames.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in filenames:
            writer.writerows([[i]])


def train_model(cfg, device, train_filepaths,
                val_filepaths, mean, std, replicate_id):

    train_dl = dataset.get_dataloader(
        cfg, train_filepaths, mean, std, augment=True)
    val_dl = dataset.get_dataloader(cfg, val_filepaths, mean, std)

    model = initialize_model(len(cfg['classes']))

    optimizer = torch.optim.AdamW(model.parameters())
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
    print(f'Rep. {replicate_id}: {epoch - 1} epochs in {minutes:.0f}m {seconds:.0f}s. Best val acc {100*best_acc:.2f}%')

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
    cfg = yaml.safe_load(open(args.config, 'r'))
    tools.set_seed(cfg, device)

    for split_id in cfg['train_splits']:

        train_fps, val_fps = dataset.get_train_filepaths(cfg, split_id)
        mean, std = dataset.get_train_data_stats(cfg, split_id)

        print(f'---------Training Model {split_id}...')
        for i in range(cfg['replicates']):
            output = train_model(cfg, device, train_fps, val_fps, mean, std, i)
            replicate_id = f'{split_id}_{i}'
            torch.save(
                output,
                os.path.join('weights', f'saved_model_{replicate_id}.pt'))
