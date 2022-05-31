# adapted from morphocluster's extract_features.py
import zipfile

import h5py
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms
from sklearn.decomposition import PCA
from tqdm import tqdm


class ArchiveDataset(torch.utils.data.Dataset):
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


def extract(path):

    # load images
    batch_size = 398 # 145668 images, 366 full batches
    transformations = transforms.Compose([transforms.ToTensor()])
    dataset = ArchiveDataset(path, transformations)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # num_workers has to be 0 because ArchiveDataset is not threadsafe
    )

    # load pretrained model and "remove" FC layer
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()

    with torch.no_grad(), h5py.File('features.h5', 'w') as f_features:

        n_objects = len(dataset)
        n_features = 32

        h5_objids = f_features.create_dataset(
            'object_id', (n_objects,), dtype=h5py.string_dtype())
        h5_features = f_features.create_dataset(
            'features', (n_objects, n_features), dtype='float32')

        offset = 0

        for objids, inputs in tqdm(loader, unit='batch'):

            features = model(inputs).numpy()
            features = PCA(n_features).fit_transform(features)

            h5_objids[offset: offset + batch_size] = objids
            h5_features[offset: offset + batch_size] = features

            offset += batch_size


if __name__ == '__main__':

    path = '/Users/particle/imgs/Archive.zip'
    path = '/home/vamaral/pico/archive.zip'
    extract(path)
