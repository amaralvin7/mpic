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

    def __init__(self, cfg, data_dir, filepaths, transformations, is_labeled=True):

        self.data_dir = data_dir
        self.filepaths = filepaths
        self.classes = sorted(cfg['classes'])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.transformations = transformations
        self.is_labeled = is_labeled
        if self.is_labeled:
            self.labels = [os.path.dirname(f) for f in filepaths]

    def __getitem__(self, index):

        filepath = os.path.join(self.data_dir, self.filepaths[index])
        if self.is_labeled:
            label = self.class_to_idx[self.labels[index]]

        with Image.open(filepath) as image:
            image = image.convert('RGB')

        image_tensor = self.transformations(image)

        if self.is_labeled:
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


def get_dataloader(cfg, data_dir, filepaths, mean, std, augment=False, is_labeled=True):

    transformations = get_transforms(cfg, mean, std, augment)
    dataloader = torch.utils.data.DataLoader(
        dataset=ParticleImages(cfg, data_dir, filepaths, transformations, is_labeled),
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['n_workers'])

    return dataloader


def calculate_data_stats(filepaths, cfg, train_split_id):
    # calculate mean and sd of dataset
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

    print(f'Calculating data stats for train split {train_split_id}...')
    dataset = ParticleImages(cfg, cfg['train_data_dir'], filepaths, get_transforms(cfg))
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


def stratified_split(cfg, domain):

    test_size = cfg['val_size']
    val_size = test_size / (1 - test_size)
    train_fps = []
    val_fps = []
    test_fps = []

    classes = [
        c for c in os.listdir(
            cfg['train_data_dir']) if os.path.isdir(
            os.path.join(
                cfg['train_data_dir'],
                c)) and c in cfg['classes']]
    for c in classes:
        filepaths = []
        for f in os.listdir(os.path.join(cfg['train_data_dir'], c)):
            ext_ok = f.split('.')[1] in cfg['exts']
            if ext_ok and re.search('.+?(?=\\d)', f).group() == domain:
                filepaths.append(os.path.join(c, f))
        c_trainval_fps, c_test_fps = train_test_split(
            filepaths, test_size=test_size, random_state=cfg['seed'])
        test_fps.extend(c_test_fps)
        c_train_fps, c_val_fps = train_test_split(
            c_trainval_fps, test_size=val_size, random_state=cfg['seed'])
        train_fps.extend(c_train_fps)
        val_fps.extend(c_val_fps)

    return train_fps, val_fps, test_fps


def write_domain_splits(cfg):

    domain_splits = {}

    for d in cfg['train_domains']:
        domain_splits[d] = {}
        train_fps, val_fps, test_fps = stratified_split(cfg, d)
        domain_splits[d]['train'] = train_fps
        domain_splits[d]['val'] = val_fps
        domain_splits[d]['test'] = test_fps

    file_path = os.path.join('..', 'data', cfg['domain_splits_fname'])
    tools.write_json(domain_splits, file_path)


def get_train_filepaths(cfg, domains):

    domain_splits = tools.load_json(os.path.join('..', 'data', cfg['domain_splits_fname']))
    train_fps = []
    val_fps = []

    for domain in domains:
        train_fps.extend(domain_splits[domain]['train'])
        val_fps.extend(domain_splits[domain]['val'])

    return train_fps, val_fps


def get_predict_filepaths(cfg, predict_domain):

    domain_splits = tools.load_json(os.path.join('..', 'data', cfg['domain_splits_fname']))
    predict_filepaths = domain_splits[predict_domain]['test']

    return predict_filepaths


def powerset(iterable):
    """https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))


def write_train_data_stats(cfg):

    data_stats = {}
    
    train_splits = powerset(cfg['train_domains'])

    for domains in train_splits:
        train_split_id = ('_').join(domains)
        train_fps, _ = get_train_filepaths(cfg, domains)
        mean, std = calculate_data_stats(train_fps, cfg, train_split_id)
        data_stats[train_split_id] = {'mean': mean.tolist(),
                                      'std': std.tolist()}

    tools.write_json(data_stats, os.path.join('..', 'data', cfg['train_data_stats_fname']))


def get_train_data_stats(cfg, train_split_id):

    train_data_stats = tools.load_json(os.path.join('..', 'data', cfg['train_data_stats_fname']))
    mean = train_data_stats[train_split_id]['mean']
    std = train_data_stats[train_split_id]['std']

    return mean, std


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(os.path.join('..', args.config), 'r'))
    tools.set_seed(cfg, 'cpu')

    write_domain_splits(cfg)
    write_train_data_stats(cfg)
