"""
Extract features from particle images.

Uses ResNet-18 pretrained on ImageNet as a feature extractor. PCA is used to
reduce feature vector dimentionality to 64. Reduced feature vectors are then
standardized and stored in a HDF5 file to be fed into morphocluster.
"""
import zipfile

import h5py
import matplotlib.pyplot as plt
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

        with self.archive.open('index.csv') as f:
            self.dataframe = pd.read_csv(
                f, dtype=str, usecols=['object_id', 'path'])

    def __getitem__(self, index):
        object_id, path = self.dataframe.iloc[index][['object_id', 'path']]

        with self.archive.open(path) as f:
            image = Image.open(f)
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return object_id, image

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
        transforms.Normalize(mean=[0.269, 0.294, 0.289],
                             std=[0.084, 0.092, 0.085]),
    ])

    dataset = ArchiveDataset(path, transformations)
    
    return dataset, len(dataset)
    
def extract(path, batch_size):
    """Extract features from an ArchiveDataset.

    Args:
        path (str): the path to the ArchiveDataset (.zip file)
        batch_size (int): batch size for the data loader
        n_components (int): number of compenents to keep in PCA

    Returns:
        image_ids (list[str]): object ids of the images
        features (numpy.ndarray): the feature matrix
    """
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
    
    features_scaled = StandardScaler().fit_transform(features)
    
    return image_ids, features, features_scaled


def pca(features, n_components):
    """PCA with standardization."""
    
    features = StandardScaler().fit_transform(features)
    features = PCA(n_components).fit_transform(features)

    return features


def write_hdf5(filename, image_ids, features, features_scaled):
    
    with h5py.File(filename, 'w') as f:
        f.create_dataset('object_id', data=image_ids, dtype=h5py.string_dtype())
        f.create_dataset('features', data=features, dtype='float32')
        f.create_dataset('features_scaled', data=features_scaled, dtype='float32')


def read_hdf5(path):
    
    with h5py.File(path, 'r') as f:
        features_unscaled = np.array(f['features'])
        features_scaled = np.array(f['features_scaled'])

    return features_unscaled, features_scaled


def cumulative_variance_plot(features_unscaled, features_scaled):

    pca_unscaled = PCA().fit(features_unscaled)
    pca_scaled = PCA().fit(features_scaled)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True, figsize=(10,4))
    x = range(1, features_scaled.shape[1] + 1)
    ax1.set_title('Unscaled features')
    ax2.set_title('Scaled features')
    ax1.set_ylabel('Cumulaive explained variance')
    ax1.plot(x, pca_unscaled.explained_variance_ratio_.cumsum(), marker='o', ls='--')
    ax2.plot(x, pca_scaled.explained_variance_ratio_.cumsum(), marker='o', ls='--')
    for ax in (ax1, ax2):
        ax.set_xlabel('Number of components')
        ax.axhline(0.8, c='k')
        ax.set_xlim(0, 100)
    
    fig.savefig('cvar')


def get_data_stats(path):
    # calculate mean and sd of dataset
    # adapted from
    # https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    batch_size = 128
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

    # path = '/Users/particle/imgs/train.zip'
    # get_data_stats(path)
    # path = '/home/vamaral/pico/train.zip'
    # image_ids, features, features_scaled = extract(path, 128)
    # write_hdf5('features_medpad_512d_scaling_comparison.h5', image_ids, features, features_scaled)

    features_unscaled, features_scaled = read_hdf5('/home/vamaral/pico/features_medpad_512d_scaling_comparison.h5')
    print(np.var(features_unscaled[:,0]), np.mean(features_unscaled[:,0]))
    print(np.var(features_scaled[:,0]), np.mean(features_scaled[:,0]))
    cumulative_variance_plot(features_unscaled, features_scaled)
