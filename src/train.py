import copy
import csv
import os

import torch

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


def train_model(cfg, device, train_filepaths, val_filepaths, mean, std, replicate_id):
    
    classes = set([f.split('/')[0] for f in train_filepaths])
    train_dl = dataset.get_dataloader(cfg, train_filepaths, classes, mean, std, augment=True)
    val_dl = dataset.get_dataloader(cfg, val_filepaths, classes, mean, std)
    model = initialize_model(len(classes))

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
    print(f'Rep. {replicate_id}: {epoch - 1} epochs in {minutes:.0f}m {seconds:.0f}s. Best val acc {100*best_acc:.2f}%')

    output = {'mean': mean,
              'std': std,
              'train_loss_hist': tl_hist,
              'train_acc_hist': ta_hist,
              'val_loss_hist': vl_hist,
              'val_acc_hist': va_hist,
              'weights': best_weights}

    return output

