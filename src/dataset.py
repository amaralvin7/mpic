import argparse
import os

import numpy as np
import torch
import yaml
from itertools import chain, combinations
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import src.tools as tools


class ParticleImages(torch.utils.data.Dataset):

    def __init__(self, cfg, filepaths, transformations, is_labeled=True):

        self.data_dir = os.path.join(cfg['data_dir'])
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


def get_transforms(cfg, mean=None, std=None, augment=False, pad=True):

    if augment:
        p = 0.5
    else:
        p = 0
    
    if pad:
        resize = CustomPad(cfg['input_size'])
    else:
        resize = transforms.Resize((cfg['input_size'], cfg['input_size']))

    transform_list = [
        resize,
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p),
        transforms.RandomHorizontalFlip(p),
        transforms.RandomVerticalFlip(p),
        transforms.ToTensor()]

    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean, std))

    transformations = transforms.Compose(transform_list)

    return transformations


def get_dataloader(cfg, filepaths, mean=None, std=None, augment=False, pad=True, is_labeled=True, shuffle=True):

    transformations = get_transforms(cfg, mean, std, augment=augment, pad=pad)
    dataloader = torch.utils.data.DataLoader(
        dataset=ParticleImages(cfg, filepaths, transformations, is_labeled),
        batch_size=cfg['batch_size'],
        shuffle=shuffle,
        num_workers=cfg['n_workers'])

    return dataloader


def calculate_data_stats(cfg, filepaths, pad):
    # calculate mean and sd of dataset
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

    print(f'Calculating data stats...')
    dataset = ParticleImages(cfg, filepaths, get_transforms(cfg, pad=pad))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['batch_size'], num_workers=0)

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


def stratified_split(classes, df, train_size, include_test):

    test_size = (1 - train_size) / 2
    val_size = test_size / (1 - test_size)
    train_fps = []
    val_fps = []
    test_fps = []

    for c in classes:
        c_df = df.loc[df['label'] == c]
        filepaths = [os.path.join(c,f) for f in c_df['filename']]
        c_trainval_fps, c_test_fps = train_test_split(filepaths, test_size=test_size, random_state=0)
        test_fps.extend(c_test_fps)
        c_train_fps, c_val_fps = train_test_split(c_trainval_fps, test_size=val_size, random_state=0)
        train_fps.extend(c_train_fps)
        val_fps.extend(c_val_fps)

    if include_test:
        return train_fps, val_fps, test_fps
    else:
        val_fps = val_fps + test_fps



def write_splits(df, filename, train_size, include_test):

    splits = {}
    domains = ['RR', 'SR', 'JC', 'FC', 'FO']

    for d in domains:
        splits[d] = {}
        d_df = df.loc[df['domain'] == d]
        classes = d_df['label'].unique()
        filepaths = stratified_split(classes, d_df, train_size, include_test)
        splits[d]['train'] = filepaths[0]
        splits[d]['val'] = filepaths[1]
        if include_test:
            splits[d]['test'] = filepaths[2]

    file_path = os.path.join('..', 'data', filename)
    tools.write_json(splits, file_path)


def compile_trainval_filepaths(cfg, domains):

    domain_splits = tools.load_json(os.path.join('..', 'data', cfg['domain_splits_fname']))
    train_fps = []
    val_fps = []

    for domain in domains:
        d_train_fps = domain_splits[domain]['train']
        d_val_fps = domain_splits[domain]['val']           
        train_fps.extend(d_train_fps)
        val_fps.extend(d_val_fps)

    return train_fps, val_fps


def compile_test_filepaths(cfg, domain):

    domain_splits = tools.load_json(os.path.join('..', 'data', cfg['domain_splits_fname']))
    test_fps = domain_splits[domain]['test']        

    return test_fps


def compile_trainvaltest_filepaths(cfg, domain):

    train_fps, val_fps = compile_trainval_filepaths(cfg, (domain,))
    test_fps = compile_test_filepaths(cfg, domain)

    return train_fps + val_fps + test_fps

def powerset(iterable):
    """https://docs.python.org/3/library/itertools.html#itertools-recipes"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))


def write_train_data_stats(cfg, write_to, pad):
    
    train_fps, _ = compile_trainval_filepaths(cfg)
    mean, std = calculate_data_stats(cfg, train_fps, pad)
    stats = {'mean': mean.tolist(), 'std': std.tolist()}

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

    tools.set_seed(0, 'cpu')

    df = tools.load_metadata()
    df = df.loc[df['label'] != 'none']

    write_splits(df, 'domain_splits.json', 0.8, True)
