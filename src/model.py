import torch
from torchvision import models


def initialize_model(n_classes, weights=None):

    if weights is not None:
        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
        model.load_state_dict(weights)
    else:  # use ImageNet weights
        model = models.resnet18(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Linear(model.fc.in_features, n_classes)

    return model
