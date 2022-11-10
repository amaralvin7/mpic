import os

import torch
from PIL import Image
from torchvision import transforms


class UnlabeledImageFolder(torch.utils.data.Dataset):

    def __init__(self, dirpath, transform):
        super().__init__()

        exts = ('jpg', 'tiff')
        filenames = os.listdir(dirpath)
        self.imagepaths = [os.path.join(dirpath, f) for f in filenames if f.split('.')[1] in exts]
        self.transform = transform

    def __getitem__(self, index):
        imagepath = self.imagepaths[index]
        filename = os.path.basename(imagepath)

        with Image.open(imagepath) as image:
            image = image.convert('RGB')

        image = self.transform(image)

        return filename, image

    def __len__(self):
        return len(self.imagepaths)


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


def get_transforms(input_size, augment=False, mean=None, std=None):
    
    if augment:
        p = 0.5
    else:
        p = 0
        
    transform_list = [
        CustomPad(input_size),
        transforms.RandomApply([transforms.RandomRotation((90,90))], p),
        transforms.RandomHorizontalFlip(p),
        transforms.RandomVerticalFlip(p),
        transforms.ToTensor()]
    
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean, std))
    
    transformations = transforms.Compose(transform_list)
    
    return transformations
