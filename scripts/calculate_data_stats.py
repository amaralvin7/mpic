import numpy as np
import torch
import yaml
from tqdm import tqdm

import src.dataset as dataset


def calculate_stats(cfg, filepaths):
    # calculate mean and sd of dataset
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

    print(f'Calculating data stats (pad={cfg["pad"]})')
    img_dataset = dataset.ParticleImages(cfg, filepaths, dataset.get_transforms(cfg))
    loader = torch.utils.data.DataLoader(
        img_dataset, batch_size=cfg['batch_size'], num_workers=cfg['n_workers'])

    sum_pixelvals = 0
    sum_square_pixelvals = 0
    n_pixels = len(img_dataset) * cfg['input_size']**2

    for pixelvals, *_ in tqdm(loader):
        sum_pixelvals += pixelvals.sum(dim=[0, 2, 3])
        sum_square_pixelvals += (pixelvals**2).sum(dim=[0, 2, 3])

    mean = sum_pixelvals / n_pixels
    var = (sum_square_pixelvals / n_pixels) - (mean**2)
    std = np.sqrt(var)

    print(f'mean = {np.around(mean.numpy(), decimals=3)}')
    print(f'std = {np.around(std.numpy(), decimals=3)}')

    return mean, std

base_cfg = yaml.safe_load(open(f'../configs/targetRR_ood.yaml', 'r'))
pad_cfg = yaml.safe_load(open(f'../configs/pad.yaml', 'r'))
train_fps = dataset.compile_filepaths(base_cfg, split='train')
for cfg in (base_cfg, pad_cfg):
    calculate_stats(cfg, train_fps)