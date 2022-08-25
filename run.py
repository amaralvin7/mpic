import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import retrain
import os

data_dir = './RR_small_tvsplit'
batch_size = 128
num_epochs = 20

val_size = 0.2
n_workers = 2

if __name__ == '__main__':
    
    mean = [0.312, 0.312, 0.320]
    std = [0.100, 0.099, 0.098]
    data_transforms = retrain.get_data_transforms(mean, std)
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=True) for x in ['train', 'val']}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = retrain.initialize_model(len(image_datasets['train'].classes))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model, train_acc, val_acc, train_loss, val_loss = retrain.train_model(
        model, dataloaders_dict, criterion, optimizer, num_epochs, device)

    
    torch.save(model.state_dict(), 'weights.pt')

    if device != 'cpu':
        train_acc = [i.cpu().numpy() for i in train_acc]
        val_acc = [i.cpu().numpy() for i in val_acc]

    plt.title("Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,num_epochs+1), train_acc, label="training")
    plt.plot(range(1,num_epochs+1), val_acc, label="validation")
    plt.legend()
    plt.savefig('accuracy')
    plt.close()

    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,num_epochs+1), train_loss, label="training")
    plt.plot(range(1,num_epochs+1), val_loss, label="validation")
    plt.legend()
    plt.savefig('loss')
    plt.close()
    
