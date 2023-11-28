import os

import torch
from PIL import Image
from torchvision import transforms
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
