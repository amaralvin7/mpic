# Adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# kept only parts relevant to using resnet18 for fine-tuning
import copy
import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets, models
from tqdm import tqdm


def train_model(model, dataloaders, criterion, optimizer, max_epochs, device):
    
    train_start = time.time()
    train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []

    epoch = 1
    best_acc = 0
    best_loss = 10**15  # an arbitrarily large number
    esi = 0  # epochs since improvement
    patience = 10

    while epoch < max_epochs and esi < patience:
        epoch_start = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                if device != 'cpu':
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):  # track history if only in train
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':  # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

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
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    # best_model_wts = copy.deepcopy(model.state_dict())
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    esi = 0
                else:
                    esi += 1
                    
        epoch_duration = time.time() - epoch_start
        print(f'Epoch {epoch} complete in {epoch_duration/60:.2f}m')
        epoch += 1 

    train_duration = time.time() - train_start
    minutes = train_duration // 60
    seconds = train_duration % 60
    print(f'Training ({epoch} epochs) complete in {minutes:.0f}m {seconds:.0f}s. Best val acc: {best_acc:4f}')

    # model.load_state_dict(best_model_wts)
    # return model, train_accuracy, val_accuracy, train_loss, val_loss
    return best_acc.cpu().numpy()


def initialize_model(num_classes):

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, num_classes)

    return model


def get_train_transforms(mean, std, input_size=128):
    
    train_transforms = transforms.Compose([
            #transforms.Resize((input_size, input_size)),
            #transforms.RandomCrop(input_size),
            transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    
    return train_transforms


def get_val_transforms(mean, std, input_size=128):
    
    val_transforms = transforms.Compose([
            #transforms.Resize((input_size, input_size)),
            #transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    
    return val_transforms


def get_data_transforms(mean, std, input_size=128):
    
    data_transforms = {'train': get_train_transforms(mean, std, input_size),
                       'val': get_val_transforms(mean, std, input_size)}
    
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


def get_data_stats(path, input_size=128, batch_size=128):
    # calculate mean and sd of dataset
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    
    print('Calculating dataset statistics...')
    transformations = transforms.Compose([#transforms.Resize((input_size, input_size)),
                                          #transforms.CenterCrop(input_size),
                                          transforms.transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transformations)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    sum_pixelvals = 0
    sum_square_pixelvals = 0
    n_pixels = len(dataset) * input_size**2

    for pixelvals, _ in tqdm(loader):
        sum_pixelvals += pixelvals.sum(dim=[0,2,3])
        sum_square_pixelvals += (pixelvals**2).sum(dim=[0,2,3])

    mean = sum_pixelvals/n_pixels
    var  = (sum_square_pixelvals/n_pixels) - (mean**2)
    std  = np.sqrt(var)

    print(f'mean = [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]')
    print(f'std = [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]')
    
    return mean, std

if __name__ == '__main__':
    
    data_dir = './RR_1p4k_tvsplit'
    batch_size = 128
    max_epochs = 200
    val_size = 0.2
    n_workers = 2
    mean, std = get_data_stats(os.path.join(data_dir, 'train'))

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
    
    best_vals = []
    
    for i in range(10):
        
        model = initialize_model(len(image_datasets['train'].classes))
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        best_vals.append(train_model(model, dataloaders_dict, criterion, optimizer, max_epochs, device))

    # model, train_acc, val_acc, train_loss, val_loss = train_model(
    #     model, dataloaders_dict, criterion, optimizer, max_epochs, device)

    # summary_plots(train_loss, val_loss, train_acc, val_acc, device)
    # save_model(model, 'weights.pt')

    print(np.mean(best_vals) * 100)
    print(np.std(best_vals) * 100)

