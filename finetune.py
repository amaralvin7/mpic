# Adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# kept only parts relevant to using resnet18 for fine-tuning
import copy
import time
import os

import matplotlib.pyplot as plt
import torch
from torchvision import transforms, datasets, models


def train_model(model, dataloaders, criterion, optimizer, max_epochs, device):
    
    train_start = time.time()
    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []

    best_acc = 0
    best_loss = 10**15  # an arbitrarily large number
    esi = 0  # epochs since improvement
    patience = 10
    epoch = 1

    while epoch < max_epochs and esi < patience:
        epoch_start = time.time()
        # print(f'Epoch {epoch} beginning...')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                if device != 'cpu':
                    inputs = inputs.to(device)
                    labels = labels.to(device)

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
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset))

            if phase == 'train':
                train_accuracy.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                val_accuracy.append(epoch_acc)
                val_loss.append(epoch_loss)
                print(f'Epoch {epoch} val loss: {epoch_loss}')
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    esi = 0
                else:
                    esi += 1
                    if esi >= patience:
                        print('***EARLY STOP***')
                    
        epoch_duration = time.time() - epoch_start
        # print(f'Epoch {epoch} complete in {epoch_duration/60:.2f}m')
        epoch += 1 

    train_duration = time.time() - train_start
    minutes = train_duration // 60
    seconds = train_duration % 60
    print(f'Training complete in {minutes:.0f}m {seconds:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, train_accuracy, val_accuracy, train_loss, val_loss


def initialize_model(num_classes):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    return model


def get_data_transforms(mean, std, input_size=128):
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    
    return data_transforms


def summary_plots(train_loss, val_loss, train_acc, val_acc, device):

    if device != 'cpu':
        train_acc = [i.cpu().numpy() for i in train_acc]
        val_acc = [i.cpu().numpy() for i in val_acc]

    num_epochs = len(train_loss)
    
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.plot(range(1, num_epochs + 1), train_acc, label='training')
    plt.plot(range(1, num_epochs + 1), val_acc, label='validation')
    plt.legend()
    plt.savefig('accuracy')
    plt.close()

    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.plot(range(1, num_epochs + 1), train_loss, label='training')
    plt.plot(range(1, num_epochs + 1), val_loss, label='validation')
    plt.legend()
    plt.savefig('loss')
    plt.close()


def save_model(model, filename):
    
    torch.save(model.state_dict(), filename)

if __name__ == '__main__':
    
    data_dir = './RR_small_tvsplit'
    batch_size = 128
    max_epochs = 500
    val_size = 0.2
    n_workers = 2
    mean = [0.317, 0.317, 0.328]
    std = [0.108, 0.108, 0.107]
    
    data_transforms = get_data_transforms(mean, std)
    image_datasets = {x: datasets.ImageFolder(
        os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x],
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True) for x in ['train', 'val']}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = initialize_model(len(image_datasets['train'].classes))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model, train_acc, val_acc, train_loss, val_loss = train_model(
        model, dataloaders_dict, criterion, optimizer, max_epochs, device)
    
    summary_plots(train_loss, val_loss, train_acc, val_acc, device)
