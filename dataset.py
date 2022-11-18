import re
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class ParticleImages(Dataset):

    def __init__(self, filepaths, transformations):

        self.filepaths = filepaths
        self.transformations = transformations
        self.labels = [os.path.basename(os.path.dirname(f)) for f in filepaths]
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __getitem__(self, index):
        
        filepath = self.filepaths[index]
        label = self.class_to_idx[self.labels[index]]

        with Image.open(filepath) as image:
            image = image.convert('RGB')

        image_tensor = self.transformations(image)

        return image_tensor, filepath, label

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
        transforms.RandomApply([transforms.RandomRotation((90,90))], p),
        transforms.RandomHorizontalFlip(p),
        transforms.RandomVerticalFlip(p),
        transforms.ToTensor()]
    
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean, std))
    
    transformations = transforms.Compose(transform_list)
    
    return transformations


def get_dataloader(cfg, filepaths, mean, std, augment=False):

    transformations = get_transforms(cfg, mean, std, augment)
    dataloader = DataLoader(
            dataset=ParticleImages(filepaths, transformations),
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['n_workers']
        )
    return dataloader


def get_data_stats(filepaths, cfg):
    # calculate mean and sd of dataset
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    
    print('Calculating dataset statistics...')
    dataset = ParticleImages(filepaths, get_transforms(cfg))
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], num_workers=0)

    sum_pixelvals = 0
    sum_square_pixelvals = 0
    n_pixels = len(dataset) * cfg['input_size']**2

    for pixelvals, *_ in tqdm(loader):
        sum_pixelvals += pixelvals.sum(dim=[0,2,3])
        sum_square_pixelvals += (pixelvals**2).sum(dim=[0,2,3])

    mean = sum_pixelvals/n_pixels
    var  = (sum_square_pixelvals/n_pixels) - (mean**2)
    std  = np.sqrt(var)

    print(f'mean = [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]')
    print(f'std = [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]')
    
    return mean, std


def stratified_split(cfg, split):
    '''Currently, only the 80/10/10 split has expected behavior'''
    domain = cfg[split]
    test_size = cfg['val_size']

    if split == 'train':
        val_size = test_size / (1 - test_size)
        all_train_fps = []
        all_val_fps = [] 
    elif split == 'test':
        all_test_fps = []
    else:
        raise ValueError('Invalid split.')

    labels = [l for l in os.listdir(cfg['data_dir']) if os.path.isdir(os.path.join(cfg['data_dir'], l)) and l not in cfg['exclude']]
    for l in labels:
        filepaths = []
        for f in os.listdir(os.path.join(cfg['data_dir'], l)):
            ext_ok = f.split('.')[1] in cfg['exts']
            if ext_ok and re.search('.+?(?=\d)', f).group() == domain:
                filepaths.append(os.path.join(cfg['data_dir'], l, f))
        trainval_fps, test_fps = train_test_split(filepaths, test_size=test_size, random_state=cfg['seed'])
        if split == 'test':
            all_test_fps.extend(test_fps)
        else:
            train_fps, val_fps = train_test_split(trainval_fps, test_size=val_size, random_state=cfg['seed'])
            all_train_fps.extend(train_fps)
            all_val_fps.extend(val_fps)
    
    if split == 'train':
        return all_train_fps, all_val_fps
    else:
        return all_test_fps

