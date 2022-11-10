
import matplotlib.pyplot as plt
import numpy as np
import torch

import dataset

path = '/Users/particle/imgs/transform_viz/'  # contains 8 images
input_size = 128
batch_size = 8

data_transforms = dataset.get_transforms(input_size, augment=True)
data = dataset.UnlabeledImageFolder(path, data_transforms)
loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

for fnames, inputs in loader:
    fig, axs = plt.subplots(1, 4)
    axs = axs.flatten()
    for j in range(len(inputs)):
        inp = inputs.data[j].numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        axs[j].axis('off')
        axs[j].imshow(inp)
        axs[j].axes.xaxis.set_visible(False)
        axs[j].axes.yaxis.set_visible(False)
    plt.show()