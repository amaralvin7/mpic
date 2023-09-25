import copy
import os

# from comet_ml import Experiment
import torch
import yaml
from itertools import product

import src.dataset as dataset
import src.tools as tools
from src.model import initialize_model
# from src.priv import comet_key


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


def train_model(cfg, exp_id, log=False):

    if log:
        experiment = Experiment(api_key=comet_key, display_summary_level=0)
        experiment.set_name(exp_id)

    print(f'----Training {exp_id}...')

    train_filepaths = dataset.compile_filepaths(cfg, cfg['train_domains'], 'train')
    val_filepaths = dataset.compile_filepaths(cfg, cfg['train_domains'], 'val')
    device = cfg['device']

    classes = set([f.split('/')[0] for f in train_filepaths])
    train_dl = dataset.get_dataloader(cfg, train_filepaths, augment=True)
    val_dl = dataset.get_dataloader(cfg, val_filepaths, augment=False)
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
        
        if log:

            stats = {'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc}
            
            experiment.log_metrics(stats, step=epoch)

        epoch += 1

    train_duration = tools.time_sync() - train_start
    minutes = train_duration // 60
    seconds = train_duration % 60
    print(f'{exp_id}: {epoch - 1} epochs in {minutes:.0f}m {seconds:.0f}s. Best val acc {100*best_acc:.2f}%')

    output = {'train_loss_hist': tl_hist,
              'train_acc_hist': ta_hist,
              'val_loss_hist': vl_hist,
              'val_acc_hist': va_hist,
              'weights': best_weights}

    torch.save(output, os.path.join('..', 'results', 'weights', f'{exp_id}.pt'))


if __name__ == '__main__':

    replicates = 5
    cfgs = [c for c in os.listdir('../configs') if '.yaml' in c]
    for cfg_fn, replicate in product(cfgs, range(replicates)):
        cfg = yaml.safe_load(open(f'../configs/{cfg_fn}', 'r'))
        tools.set_seed(replicate)
        exp_id = f'{cfg_fn.split(".")[0]}-{replicate}'
        train_model(cfg, exp_id)
    