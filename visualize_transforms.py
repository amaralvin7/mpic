
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# this is a folder with 8 images in each class (aggregates, copepods)
path = '/Users/particle/imgs/labeled_grouped_ttsplit/RR_supersmall_tvsplit/train/'
input_size = 128

mean = [0.303, 0.298, 0.300]
std = [0.104, 0.103, 0.104]

data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomApply([transforms.RandomRotation((90,90))], p=0.5),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor()])
            #transforms.Normalize(mean, std)])

data = datasets.ImageFolder(path, data_transforms)
loader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False)

for i, (inputs, labels) in enumerate(loader):
    
    fig, axs = plt.subplots(2,4)
    axs = axs.flatten()
    for j in range(inputs.size()[0]):
        inp = inputs.data[j].numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        axs[j].imshow(inp)
        axs[j].axes.xaxis.set_visible(False)
        axs[j].axes.yaxis.set_visible(False)
    plt.show()