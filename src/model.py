import torch
from torchvision import models


def initialize_model(model_arc, n_classes, weights=None):

    if model_arc == 'resnet18':
        if weights is not None:
            model = models.resnet18()
            model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
            model.load_state_dict(weights)
        else:  # use ImageNet weights
            model = models.resnet18(pretrained=True)  # for older versions of torch
            model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    else:
        if weights is not None:
            model = models.resnet50()
            model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
            model.load_state_dict(weights)
        else:
            model = models.resnet50(pretrained=True)  # for older versions of torch
            model.fc = torch.nn.Linear(model.fc.in_features, n_classes)

    return model
