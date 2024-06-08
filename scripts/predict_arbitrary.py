import os
import yaml

import torch
from PIL import Image

import src.dataset
import src.predict
import src.tools


class UnlabeledImages(torch.utils.data.Dataset):

    def __init__(self, filepaths, transformations):

        self.filepaths = filepaths
        self.classes = sorted(cfg['classes'])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for i, c in enumerate(self.classes)}
        self.transformations = transformations

    def __getitem__(self, index):
        
        filepath = self.filepaths[index]

        with Image.open(filepath) as image:
            image = image.convert('RGB')

        image_tensor = self.transformations(image)

        return image_tensor, filepath

    def __len__(self):
        return len(self.filepaths)


def get_dataloader(cfg, filepaths):

    transformations = src.dataset.get_transforms(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset=UnlabeledImages(filepaths, transformations),
        batch_size=cfg['batch_size'],
        num_workers=cfg['n_workers'])

    return dataloader


if __name__ == '__main__':

    data_path = input('Enter data path: ')
    
    model_name = 'targetUN_ood'
    cfg = yaml.safe_load(open(f'../configs/{model_name}.yaml', 'r'))
    models = sorted([m for m in os.listdir(f'../results/weights/') if model_name in m])
    for m in models:
        model_id = m.split('.')[0]
        fps = [os.path.join(data_path, f) for f in os.listdir(data_path) if f[0] != '.']
        loader = get_dataloader(cfg, sorted(fps))
        src.predict.predict_labels(cfg['device'], loader, model_id)
    