
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

# this is a folder with 8 images in each class (aggregates, copepods)
path = '/Users/particle/imgs/labeled_grouped_ttsplit/RR_supersmall_tvsplit/train/'
input_size = 128

data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size),
            # transforms.RandomResizedCrop(input_size),
            transforms.ToTensor()])

data = datasets.ImageFolder(path, data_transforms)
loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False)

for i, (inputs, labels) in enumerate(loader):
    
    fig, axs = plt.subplots(2,4)
    axs = axs.flatten()
    for j in range(inputs.size()[0]):
        inp = inputs.data[j].numpy().transpose((1, 2, 0))
        axs[j].imshow(inp)
        axs[j].axes.xaxis.set_visible(False)
        axs[j].axes.yaxis.set_visible(False)
    plt.show()