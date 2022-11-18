# Adapted from https://github.com/CV4EcologySchool/ct_classifier/blob/master/ct_classifier/train.py
# and https://github.com/CV4EcologySchool/Lecture-13/blob/main/cv4e_lecture13/train.py
# and https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

import argparse
import copy
import sys

import matplotlib.pyplot as plt
import torch
import yaml
from tqdm import trange

import dataset
import tools
from model import initialize_model


def training_epoch(device, dataloader, model, optimizer, criterion, update):
    '''
        Our actual training function.
    '''
    torch.set_grad_enabled(update)
    model.to(device)

    if update:
        model.train()
        phase = 'Train'
    else:
        model.eval()
        phase = 'Val'

    # running averages
    loss_total = 0.0
    acc_total = 0.0

    progress = trange(len(dataloader))
        
    for i, (inputs, _, labels) in enumerate(dataloader):

        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if update:
            optimizer.zero_grad()  # check if it matters if this is before loss calculation
            loss.backward()
            optimizer.step()
            
        loss_total += loss.item()  # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor
        predictions = torch.argmax(outputs, dim=1)    # the predicted label is the one at position (class index) with highest predicted value
        acc_total += torch.mean((predictions == labels).float()).item() # number of correct predictions divided by batch size (i.e., average/mean)

        progress.set_description(
            f'[{phase}] Loss: {loss_total/(i+1):.2f}; Acc: {100*acc_total/(i+1):.2f}%')
        progress.update(1)
    
    # end of epoch; finalize
    progress.close()
    loss_total /= len(dataloader)
    acc_total /= len(dataloader)

    return loss_total, acc_total


def test_splits(train_filepaths, val_filepaths):
    
    print('---TRAIN---')
    for i in train_filepaths:
        print(i)
    print('---VAL---')
    for i in val_filepaths:
        print(i)
    # print('---TEST---')
    # for i in results2:
    #     print(i)


def train_model(cfg, device):

    train_filepaths, val_filepaths =  dataset.stratified_split(cfg, 'train')
    mean, std = dataset.get_data_stats(train_filepaths, cfg)
    train_dl = dataset.get_dataloader(cfg, train_filepaths, mean, std, augment=True)
    val_dl = dataset.get_dataloader(cfg, val_filepaths, mean, std)

    model = initialize_model(len(train_dl.dataset.classes), device)

    optimizer = torch.optim.Adam(model.parameters())
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
        
        epoch_start = tools.time_sync()

        train_loss, train_acc = training_epoch(device, train_dl, model, optimizer, criterion, True)
        val_loss, val_acc = training_epoch(device, val_dl, model, optimizer, criterion, False)
        
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

        epoch_duration = tools.time_sync() - epoch_start
        print(f'Epoch {epoch} complete in {epoch_duration/60:.2f}m')
        epoch += 1

    train_duration = tools.time_sync() - train_start
    minutes = train_duration // 60
    seconds = train_duration % 60
    print(f'{epoch - 1} epochs complete in {minutes:.0f}m {seconds:.0f}s. Best val acc: {100*best_acc:.2f}%')
    
    output = {'mean': mean,
              'std': std,
              'train_loss_hist': tl_hist,
              'train_acc_hist': ta_hist,
              'val_loss_hist': vl_hist,
              'val_acc_hist': va_hist,
              'weights': best_weights}
    
    return output


def summary_plots(tl_hist, vl_hist, ta_hist, va_hist):

    num_epochs = len(tl_hist)
    
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.plot(range(1, num_epochs + 1), ta_hist, label='training')
    plt.plot(range(1, num_epochs + 1), va_hist, label='validation')
    plt.legend()
    plt.savefig('accuracy')
    plt.close()

    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.plot(range(1, num_epochs + 1), tl_hist, label='training')
    plt.plot(range(1, num_epochs + 1), vl_hist, label='validation')
    plt.legend()
    plt.savefig('loss')
    plt.close()


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config', default='config.yaml')
    # args = parser.parse_args()

    # cfg = yaml.safe_load(open(args.config, 'r'))
    
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    
    tools.set_seed(cfg, device)
    
    output = train_model(cfg, device)

    torch.save(output, 'saved_model.pt')
    
    summary_plots(
        output['train_loss_hist'], output['val_loss_hist'],
        output['train_acc_hist'], output['val_acc_hist'])

