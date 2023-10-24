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


def stratified_split(classes, df, train_size, include_test):

    def get_class_df(particle_class):

        c_df = df.loc[df['label'] == particle_class]
        filepaths = [os.path.join(c,f) for f in c_df['filename']]

        return filepaths

    train_fps = []
    val_fps = []

    if include_test:
        test_size = (1 - train_size) / 2
        val_size = test_size / (1 - test_size)
        test_fps = []
        for c in classes:
            filepaths = get_class_df(c)
            c_trainval_fps, c_test_fps = train_test_split(filepaths, test_size=test_size, random_state=0)
            test_fps.extend(c_test_fps)
            c_train_fps, c_val_fps = train_test_split(c_trainval_fps, test_size=val_size, random_state=0)
            train_fps.extend(c_train_fps)
            val_fps.extend(c_val_fps)
        return train_fps, val_fps, test_fps
    else:
        val_size = 1 - train_size
        for c in classes:
            filepaths = get_class_df(c)
            c_train_fps, c_val_fps = train_test_split(filepaths, test_size=val_size, random_state=0)
            train_fps.extend(c_train_fps)
            val_fps.extend(c_val_fps)
        return train_fps, val_fps


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


def compile_filepaths(cfg, domains, split):

    splits = tools.load_json(os.path.join('..', 'data', cfg['splits_fname']))
    fps = []

    for d in domains:
        d_fps = splits[d][split]         
        fps.extend(d_fps)
    
    return fps


def write_splits_hitloopII():

    def get_splits_dict(image_dir):

        splits_dict = tools.load_json('../data/splits_hitloopI.json')
        classes = os.listdir(image_dir)
        df_rows = []
        for c in classes:
            filenames = [f for f in os.listdir(f'{image_dir}/{c}') if 'RR' in f]
            for f in filenames:
                df_rows.append({'filename': f, 'label': c})
        labeled_df = pd.DataFrame(df_rows)
        train_fps, val_fps = stratified_split(classes, labeled_df, 0.8, False)  # stratify based on new labels
        test_df = metadata.merge(labeled_df, on='filename', how='left', indicator=True)  # all RR images not in folder
        test_df = test_df.loc[test_df['_merge'] == 'left_only']
        test_df['filepath'] = test_df['label_x'] + '/' + test_df['filename']
        splits_dict['RR'] = {'train': train_fps, 'val': val_fps, 'test': list(test_df['filepath'])}

        for fp in list(test_df['filepath']):
            os.makedirs(f'{image_dir}/{fp.split("/")[0]}', exist_ok=True)
            shutil.copyfile(f'../../mpic_data/imgs/{fp}', f'{image_dir}/{fp}')

        return splits_dict

    metadata = tools.load_metadata()
    metadata = metadata.loc[metadata['domain'] == 'RR'][['filename', 'label']]
    replicates = 5
    img_dirs = {'C': '../../mpic_data/imgs_fromB',
                'D': '../../mpic_data/imgs_fromB_verified',
                'E': '../../mpic_data/imgs_fromB_voted200',
                'F': '../../mpic_data/imgs_fromB_voted400'}

    for model, i in product(('C', 'D'), range(replicates)):
        splits_dict = get_splits_dict(f'{img_dirs[model]}/{i}')
        tools.write_json(splits_dict, f'../data/splits_hitloopII_{model}-{i}.json')

    for model in ('E', 'F'):
        splits_dict = get_splits_dict(img_dirs[model])
        tools.write_json(splits_dict, f'../data/splits_hitloopII_{model}.json')


if __name__ == '__main__':

    df = tools.load_metadata()
    df = df.loc[df['label'] != 'none']

    # #hyperparameter tuning (hptune) experiments
    # write_splits(df, 'splits.json', 0.8, True)

    # base_cfg = yaml.safe_load(open(f'../configs/hptune/base.yaml', 'r'))
    # pad_cfg = yaml.safe_load(open(f'../configs/hptune/pad.yaml', 'r'))
    # train_fps = compile_filepaths(base_cfg, base_cfg['train_domains'], split='train')
    # for cfg in (base_cfg, pad_cfg):
    #     calculate_data_stats(cfg, train_fps)
    
    # #human-in-the-loop I (hitloopI) experiments
    # write_splits(df, 'splits_hitloopI.json', 0.8, False)

    #human-in-the-loop II (hitloopII) experiments
    write_splits_hitloopII()