import os
from random import sample, seed
import shutil

path = './data7by4k/'
contents = os.listdir(path)
labels = [name for name in contents if os.path.isdir(f'{path}{name}')]
n_smps = 4000
seed(0)

for l in labels:
    filenames = [f for f in os.listdir(f'./data/by_label/{l}') if f.endswith('.jpg')]
    subsample = sample(filenames, n_smps)
    for s in subsample:
        shutil.copy(f'./data/by_label/{l}/{s}', f'{path}{l}')