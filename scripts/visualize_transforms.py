import os
import random

import matplotlib.pyplot as plt
import numpy as np
import yaml

import src.dataset as dataset

cfg = yaml.safe_load(open('../config.yaml', 'r'))

filepaths =['aggregate/JC21_7x_obl_12_24999.tiff',
            'bubble/JC8_32x_obl_6b_24105.tiff',
            'long_pellet/JC56_7x_obl_2_12066.tiff',
            'noise/JC6_32x_obl_35c_49889.tiff']

loaders = [dataset.get_dataloader(cfg, filepaths,
                                  mean=None, std=None, augment=False,
                                  pad=b, shuffle=False) for b in (True, False)]

fig, axs = plt.subplots(4, 2)

for j, l in enumerate(loaders):
    for inputs, filepaths, _ in l:
        for i in range(len(inputs)):
            inp = inputs.data[i].numpy().transpose((1, 2, 0))
            inp = np.clip(inp, 0, 1)
            # axs[i,j].axis('off')
            axs[i,j].imshow(inp)
            axs[i,j].axes.xaxis.set_visible(False)
            axs[i,j].axes.yaxis.set_visible(False)
            axs[i,j].spines[['top', 'right', 'bottom', 'left']].set_visible(False)
axs[0,0].set_title('CustomPad')
axs[0,1].set_title('Resize')
run_folder = 'pad_exp_targetRR'
fig.savefig(f'../runs/{run_folder}/figs/visualize_transforms.png')
