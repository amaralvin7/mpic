"""
Extract features from particle images.

Uses ResNet-18 pretrained on ImageNet as a feature extractor. PCA is used to
reduce feature vector dimentionality to 64. Reduced feature vectors are then
standardized and stored in a HDF5 file to be fed into morphocluster.
"""
import zipfile

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from torchvision import models, transforms
from tqdm import tqdm

class ArchiveDataset(torch.utils.data.Dataset):
    # from morphocluster
    # https://github.com/morphocluster/morphocluster/blob/0.2.x/morphocluster/processing/extract_features.py
    def __init__(self, archive_fn: str, transform=None):
        super().__init__()

        self.transform = transform
        self.archive = zipfile.ZipFile(archive_fn)

        with self.archive.open('index.csv') as fp:
            self.dataframe = pd.read_csv(
                fp, dtype=str, usecols=['object_id', 'path'])

    def __getitem__(self, index):
        object_id, path = self.dataframe.iloc[index][['object_id', 'path']]

        with self.archive.open(path) as fp:
            img = Image.open(fp)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return object_id, img

    def __len__(self):
        return len(self.dataframe)


def get_data_loader(dataset, batch_size):

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # num_workers has to be 0 because ArchiveDataset is not threadsafe
    )
    
    return loader

def load_dataset(path):

    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2112, 0.2303, 0.2232],
                             std=[0.1407, 0.1532, 0.1467]),
    ])

    dataset = ArchiveDataset(path, transformations)
    
    return dataset, len(dataset)
    
def extract(path, batch_size, n_components):

    dataset, n_images = load_dataset(path)
    loader = get_data_loader(dataset, batch_size)

    # load pretrained model and "remove" FC layer
    model = models.resnet18(pretrained=True)
    n_features = model.fc.in_features
    model.fc = torch.nn.Identity()
    model.eval()

    features = np.empty((n_images, n_features))
    image_ids = []

    with torch.no_grad():
        i = 0
        for ids, inputs in tqdm(loader, unit='batch'):
            image_ids.extend(ids)
            feature_vectors = model(inputs).numpy()
            features[i:i + batch_size] = feature_vectors
            i += batch_size
    
    features = pca(features, n_components)
    features = StandardScaler().fit_transform(features)  # rescale for clustering
    
    return image_ids, features


def pca(features, n_components):

    features = StandardScaler().fit_transform(features)
    features = PCA(n_components).fit_transform(features)

    return features


def write_hdf5(filename, image_ids, features):
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('object_id', data=image_ids, dtype=h5py.string_dtype())
        f.create_dataset('features', data=features, dtype='float32')


def get_data_stats():
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    batch_size = 64
    transformations = transforms.Compose([transforms.ToTensor()])
    dataset = ArchiveDataset(path, transformations)
    loader = get_data_loader(dataset, batch_size)

    sum_pixelvals = 0
    sum_square_pixelvals = 0
    n_pixels = len(dataset) * 224**2

    for _, pixelvals in tqdm(loader):
        sum_pixelvals += pixelvals.sum(dim=[0,2,3])
        sum_square_pixelvals += (pixelvals**2).sum(dim=[0,2,3])

    mean = sum_pixelvals/n_pixels
    var  = (sum_square_pixelvals/n_pixels) - (mean**2)
    sd  = torch.sqrt(var)

    print(f'mean: {mean}')
    print(f'sd: {sd}')

if __name__ == '__main__':

    # path = '/Users/particle/imgs/archive.zip'
    path = '/home/vamaral/pico/archive.zip' 
    image_ids, features = extract(path, 128, 64)
    write_hdf5('features.h5', image_ids, features)

