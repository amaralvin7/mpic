# Adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# kept only parts relevant to using resnet18 for fine-tuning
from __future__ import print_function
from __future__ import division
import time
import copy

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    train_start = time.time()

    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch + 1}/{num_epochs} beginning...')

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

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double()
                         / len(dataloaders[phase].dataset))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_accuracy.append(epoch_acc)
                val_loss.append(epoch_loss)
            else:
                train_accuracy.append(epoch_acc)
                train_loss.append(epoch_loss)

        epoch_duration = time.time() - epoch_start
        print(f'Epoch {epoch + 1} complete in {epoch_duration//60:.0f}m')

    train_duration = time.time() - train_start
    print(f'Training complete in {train_duration//3600:.0f}h {train_duration%60:.0f}m')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_accuracy, val_accuracy, train_loss, val_loss

def initialize_model(num_classes):

    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    return model

def get_transforms(mean, std):

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),  # specified in https://pytorch.org/hub/pytorch_vision_inception_v3/
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    
    return transformations

def get_dataloaders(train_data, val_data, test_data, batch_size, n_workers):

    train_val_datasets = {'train': train_data, 'val': val_data}
    train_loaders = {k: DataLoader(
        v, batch_size=batch_size, shuffle=True, num_workers=n_workers)
                   for k, v in train_val_datasets.items()}
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    
    return train_loaders, test_loader

def train_val_test_split(data, test_size):
    
    indices = list(range(len(data)))

    train_val_i, test_i, train_val_labels, _ = train_test_split(
        indices, data.targets, test_size=test_size, stratify=data.targets)

    new_test_size = test_size/(1 - test_size)
    train_i, val_i, _, _ = train_test_split(
        train_val_i, train_val_labels, test_size=new_test_size,
        stratify=train_val_labels)
    
    train_data = Subset(data, indices=train_i)
    val_data = Subset(data, indices=val_i)
    test_data = Subset(data, indices=test_i)

    return train_data, val_data, test_data
    
