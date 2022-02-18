# Adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# kept only parts relevant to using inception v3 as a feature extractor

import time
import copy

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    train_acc_history = []
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for _ in tqdm(range(num_epochs), desc='Epoch'):
        # print(f'Epoch {epoch}/{num_epochs - 1}')
        # print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double()
                         / len(dataloaders[phase].dataset))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.cpu().numpy())
            else:
                train_acc_history.append(epoch_acc.cpu().numpy())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history

def initialize_model(num_classes):
    """ Inception v3
    Be careful, expects (299,299) sized images and has auxiliary output
    """
    model = models.inception_v3(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    # Handle the auxilary net
    num_features_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_features_aux, num_classes)
    
    # Handle the primary net
    num_features_main = model.fc.in_features
    model.fc = nn.Linear(num_features_main, num_classes)

    return model

