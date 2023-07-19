import argparse
import re
import os
import sys

import numpy as np
import torch
import yaml
from itertools import chain, combinations
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split

import src.tools as tools


class ParticleImages(torch.utils.data.Dataset):

    def __init__(self, cfg, filepaths, classes, transformations, is_labeled=True):

        self.data_dir = os.path.join(cfg['data_dir'], 'images')
        self.filepaths = filepaths
        self.classes = sorted(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.transformations = transformations
        self.is_labeled = is_labeled
        if self.is_labeled:
            self.labels = [os.path.dirname(f) for f in filepaths]

    def __getitem__(self, index):

        filepath = os.path.join(self.data_dir, self.filepaths[index])

        with Image.open(filepath) as image:
            image = image.convert('RGB')

        image_tensor = self.transformations(image)

        if self.is_labeled:
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


def get_transforms(cfg, mean=None, std=None, augment=False):

    if augment:
        p = 0.5
    else:
        p = 0

    transform_list = [
        CustomPad(cfg['input_size']),
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p),
        transforms.RandomHorizontalFlip(p),
        transforms.RandomVerticalFlip(p),
        transforms.ToTensor()]

    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean, std))

    transformations = transforms.Compose(transform_list)

    return transformations


def get_dataloader(cfg, filepaths, classes, mean, std, augment=False, is_labeled=True):

    transformations = get_transforms(cfg, mean, std, augment)
    dataloader = torch.utils.data.DataLoader(
        dataset=ParticleImages(cfg, filepaths, classes, transformations, is_labeled),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['n_workers'])

    return dataloader


def calculate_data_stats(cfg, filepaths, classes):
    # calculate mean and sd of dataset
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

    print(f'Calculating data stats...')
    dataset = ParticleImages(cfg, filepaths, classes, get_transforms(cfg))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['batch_size'], num_workers=0)

    sum_pixelvals = 0
    sum_square_pixelvals = 0
    n_pixels = len(dataset) * cfg['input_size']**2

    for pixelvals, *_ in loader:
        sum_pixelvals += pixelvals.sum(dim=[0, 2, 3])
        sum_square_pixelvals += (pixelvals**2).sum(dim=[0, 2, 3])

    mean = sum_pixelvals / n_pixels
    var = (sum_square_pixelvals / n_pixels) - (mean**2)
    std = np.sqrt(var)

    print(f'mean = {np.around(mean.numpy(), decimals=3)}')
    print(f'std = {np.around(std.numpy(), decimals=3)}')

    return mean, std


def stratified_split(cfg, df):

    test_size = cfg['val_size']
    val_size = test_size / (1 - test_size)
    train_fps = []
    val_fps = []
    test_fps = []

    classes = df['label'].unique()
    
    for c in classes:
        c_df = df.loc[df['label'] == c]
        filepaths = [os.path.join(c,f) for f in c_df['filename']]
        c_trainval_fps, c_test_fps = train_test_split(
            filepaths, test_size=test_size, random_state=cfg['seed'])
        test_fps.extend(c_test_fps)
        c_train_fps, c_val_fps = train_test_split(
            c_trainval_fps, test_size=val_size, random_state=cfg['seed'])
        train_fps.extend(c_train_fps)
        val_fps.extend(c_val_fps)

    return train_fps, val_fps, test_fps


def write_domain_splits(cfg, df):

    domain_splits = {}

    for d in cfg['domains']:
        domain_splits[d] = {}
        d_df = df.loc[df['domain'] == d]
        train_fps, val_fps, test_fps = stratified_split(cfg, d_df)
        domain_splits[d]['train'] = train_fps
        domain_splits[d]['val'] = val_fps
        domain_splits[d]['test'] = test_fps

    file_path = os.path.join('..', 'data', cfg['domain_splits_fname'])
    tools.write_json(domain_splits, file_path)


def compile_domain_filepaths(cfg, domains, classes=None):

    domain_splits = tools.load_json(os.path.join('..', 'data', cfg['domain_splits_fname']))
    train_fps = []
    val_fps = []

    for domain in domains:
        if classes:
            d_train_fps = [f for f in domain_splits[domain]['train'] if f.split('/')[0] in classes]
            d_val_fps = [f for f in domain_splits[domain]['val'] if f.split('/')[0] in classes]
        else:
            d_train_fps = domain_splits[domain]['train']
            d_val_fps = domain_splits[domain]['val']           
        train_fps.extend(d_train_fps)
        val_fps.extend(d_val_fps)

    return train_fps, val_fps


def powerset(iterable):
    """https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))


def write_train_data_stats(cfg, write_to, for_ablations=False):
    
    def filepaths_to_data_stats(filepaths):

        mean, std = calculate_data_stats(cfg, filepaths, classes)
        stats_dict = {'mean': mean.tolist(), 'std': std.tolist()}
        
        return stats_dict
        
    if for_ablations:
        classes = cfg['ablation_classes']
        stats = {}
        train_splits = powerset(cfg['domains'])
        for domains in train_splits:
            train_split_id = ('_').join(domains)
            train_fps, _ = compile_domain_filepaths(cfg, domains, classes)
            stats[train_split_id] = filepaths_to_data_stats(train_fps)
    else:
        classes = cfg['all_classes']
        train_fps, _ = compile_domain_filepaths(cfg, cfg['domains'])
        stats = filepaths_to_data_stats(train_fps)

    tools.write_json(stats, os.path.join('..', 'data', write_to))


def get_data_stats(train_split_id=None):
    
    data_stats_fname = 'data_stats_ablations.json' if train_split_id else 'data_stats.json'
    train_data_stats = tools.load_json(os.path.join('..', 'data', data_stats_fname))

    if train_split_id:
        mean = train_data_stats[train_split_id]['mean']
        std = train_data_stats[train_split_id]['std']
    else:
        mean = train_data_stats['mean']
        std = train_data_stats['std']       

    return mean, std


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(os.path.join('..', args.config), 'r'))
    tools.set_seed(cfg, 'cpu')

    df = tools.load_metadata(cfg)
    df = df.loc[df['label'] != 'none']

    write_domain_splits(cfg, df)
    # write_train_data_stats(cfg, 'data_stats.json', False)
    # write_train_data_stats(cfg, 'data_stats_ablations.json', True)
