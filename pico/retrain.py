import os
import torch
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Lambda

labels = [name for name in os.listdir(f'./data/by_label/') if os.path.isdir(f'./data/by_label/{name}')]
n_labels = len(labels)

transform = ToTensor()
target_transform = Lambda(
    lambda y: torch.zeros(
        n_labels, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y),
                                              value=1))

dataset = datasets.DatasetFolder('./data/by_label/',
                                 loader = datasets.folder.default_loader,
                                 extensions = ('.jpg',),
                                 transform=transform,
                                 target_transform=target_transform)

print(dataset[0])

                                 