import os
import shutil
import sys
from itertools import product

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import src.tools as tools


class ParticleImages(torch.utils.data.Dataset):

    def __init__(self, cfg, filepaths, transformations, train=True):

        self.data_dir = os.path.join(cfg['data_dir'])
        self.filepaths = filepaths  # <label>/<filename>
        self.classes = sorted(cfg['classes'])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.transformations = transformations
        self.train = train
        if self.train:
            self.labels = [os.path.dirname(f) for f in filepaths]

    def __getitem__(self, index):
        
        filepath = self.filepaths[index]

        with Image.open(f'{self.data_dir}/{filepath}') as image:
            image = image.convert('RGB')

        image_tensor = self.transformations(image)

        if self.train:
            label = self.class_to_idx[self.labels[index]]
            return image_tensor, filepath, label
        return image_tensor, filepath

    def __len__(self):
        return len(self.filepaths)


class CustomPad:
    """Rescale and center images along the longest axis, then zero-pad.

    Adapted from:
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

    Args:
        image (PIL.Image.Image): image to be padded

    Returns:
        padded_image (PIL.Image.Image): padded image
    """

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, image):
        max_dim = max(image.size)
        if max_dim > self.input_size:  # rescale if image is larger than square
            ratio = self.input_size / max_dim
            scaled_size = [int(x * ratio) for x in image.size]
            image = image.resize(scaled_size)
        else:
            scaled_size = image.size

        padded_image = Image.new('RGB', (self.input_size, self.input_size))
        paste_at = [(self.input_size - s) // 2 for s in scaled_size]
        padded_image.paste(image, paste_at)

        return padded_image


def get_transforms(cfg, augment=False):

    if augment:
        p = 0.5
    else:
        p = 0
    
    if cfg['pad']:
        resize = CustomPad(cfg['input_size'])
    else:
        resize = transforms.Resize((cfg['input_size'], cfg['input_size']))

    transform_list = [
        resize,
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p),
        transforms.RandomHorizontalFlip(p),
        transforms.RandomVerticalFlip(p),
        transforms.ToTensor()]

    if cfg['mean'] is not None and cfg['std'] is not None:
        transform_list.append(transforms.Normalize(cfg['mean'], cfg['std']))

    transformations = transforms.Compose(transform_list)

    return transformations


def get_dataloader(cfg, filepaths, augment=False, train=True, shuffle=True):

    transformations = get_transforms(cfg, augment=augment)
    dataloader = torch.utils.data.DataLoader(
        dataset=ParticleImages(cfg, filepaths, transformations, train),
        batch_size=cfg['batch_size'],
        shuffle=shuffle,
        num_workers=cfg['n_workers'])

    return dataloader


def calculate_data_stats(cfg, filepaths):
    # calculate mean and sd of dataset
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

    print(f'Calculating data stats (pad={cfg["pad"]}, train_domains={cfg["train_domains"]})')
    dataset = ParticleImages(cfg, filepaths, get_transforms(cfg))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['batch_size'], num_workers=cfg['n_workers'])

    sum_pixelvals = 0
    sum_square_pixelvals = 0
    n_pixels = len(dataset) * cfg['input_size']**2

    for pixelvals, *_ in tqdm(loader):
        sum_pixelvals += pixelvals.sum(dim=[0, 2, 3])
        sum_square_pixelvals += (pixelvals**2).sum(dim=[0, 2, 3])

    mean = sum_pixelvals / n_pixels
    var = (sum_square_pixelvals / n_pixels) - (mean**2)
    std = np.sqrt(var)

    print(f'mean = {np.around(mean.numpy(), decimals=3)}')
    print(f'std = {np.around(std.numpy(), decimals=3)}')

    return mean, std


def stratified_split(df, train_size):

    def get_class_df(particle_class):

        c_df = df.loc[df['label'] == particle_class]
        filepaths = [os.path.join(c,f) for f in c_df['filename']]

        return filepaths

    train_fps = []
    val_fps = []
    classes = df['label'].unique()

    val_size = 1 - train_size
    for c in classes:
        filepaths = get_class_df(c)
        c_train_fps, c_val_fps = train_test_split(filepaths, test_size=val_size, random_state=0)
        train_fps.extend(c_train_fps)
        val_fps.extend(c_val_fps)
    return train_fps, val_fps


def compile_filepaths(cfg, split):

    splits = tools.load_json(os.path.join('..', 'data', cfg['splits_fname']))
    fps = splits[split]
    
    return fps


if __name__ == '__main__':

    df = tools.load_metadata()
    df = df.loc[df['label'] != 'none']

    # #hyperparameter tuning (hptune) experiments
    # write_splits(df, 'splits.json', 0.8, ['RR', 'SR', 'JC', 'FC', 'FO'], True)

    # base_cfg = yaml.safe_load(open(f'../configs/hptune/base.yaml', 'r'))
    # pad_cfg = yaml.safe_load(open(f'../configs/hptune/pad.yaml', 'r'))
    # train_fps = compile_filepaths(base_cfg, base_cfg['train_domains'], split='train')
    # for cfg in (base_cfg, pad_cfg):
    #     calculate_data_stats(cfg, train_fps)
