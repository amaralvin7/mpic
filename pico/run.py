"""
todo
- split by train/val/test
- stratify splits
- data augmentation
- normalize specifically to my dataset
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import retrain

# # Detect if we have a GPU available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16
num_workers = 0
num_epochs = 100
input_size = 299

# Define the preprocessing transforms
# https://pytorch.org/hub/pytorch_vision_inception_v3/
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),])

# load the full dataset
data_dir = "./smalldata"
full_dataset = datasets.ImageFolder(data_dir, preprocess)
print(full_dataset.classes)

# train/val split (non-stratified)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size])

# create dataloaders for the train/val datasets
img_datasets = {'train': train_dataset, 'val': val_dataset}
dataloaders_dict = {k: DataLoader(
    v, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                    for k, v in img_datasets.items()}

# initialize the model
num_classes = len(full_dataset.classes)
model = retrain.initialize_model(num_classes)
# model = model.to(device)

# figure out which params we have to update
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = optim.Adam(params_to_update)
criterion = nn.CrossEntropyLoss()

#retrain the model
model, vhist, thist = retrain.train_model(
    model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)

# plot training and validation accuracy histories
plt.title("Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.plot(range(1,num_epochs+1),vhist,label="validation")
plt.plot(range(1,num_epochs+1),thist,label="training")
plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.ylim([-0.1, 1.1])
plt.legend()
plt.savefig('history')
plt.close()
    
