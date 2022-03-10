"""
todo
- data augmentation
- normalize specifically to my dataset
"""
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
import retrain

# # Detect if we have a GPU available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data_path = './data/by_label'
# batch_size = 64
# n_epochs = 20

# data_path = './smalldata'
# batch_size = 2
# n_epochs = 5

data_path = './ants_bees'
batch_size = 8
n_epochs = 15

test_size = 0.1
n_workers = 0

if __name__ == '__main__':
    
    transforms = retrain.get_transforms()
    data = datasets.ImageFolder(data_path, transforms)
    train_data, val_data, test_data = retrain.train_val_test_split(
        data, test_size)
    train_loaders, test_loader = retrain.get_dataloaders(
        train_data, val_data, test_data, batch_size, n_workers)

    model = retrain.initialize_model(len(data.classes))
    # model = model.to(device)
    params_to_update = retrain.get_params_to_update(model)
    # optimizer = optim.Adam(params_to_update)
    optimizer = torch.optim.SGD(params_to_update, lr=0.05, momentum=0.0)
    criterion = torch.nn.CrossEntropyLoss()
    model, train_acc, val_acc, train_loss, val_loss = retrain.train_model(
        model, train_loaders, criterion, optimizer, num_epochs=n_epochs)

    pkl = (train_acc, val_acc, train_loss, val_loss, test_loader, data)
    with open('modeldata.pkl', 'wb') as file:
                pickle.dump(pkl, file)
    
    torch.save(model.state_dict(), 'weights.pt')

    # plot training and validation accuracy histories
    plt.title("Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,n_epochs+1), train_acc, label="training")
    plt.plot(range(1,n_epochs+1), val_acc, label="validation")
    plt.xticks(np.arange(1, n_epochs+1, 1.0))
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.savefig('accuracy')
    plt.close()

    # plot training and validation loss histories
    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,n_epochs+1), train_loss, label="training")
    plt.plot(range(1,n_epochs+1), val_loss, label="validation")
    plt.xticks(np.arange(1, n_epochs+1, 1.0))
    plt.legend()
    plt.savefig('loss')
    plt.close()
    
