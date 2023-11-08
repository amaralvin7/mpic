import os
import shutil
from itertools import product

# must copy non RR images into the directories with the images predicted from B

replicates = 5
copy_from = '../../mpic_data/imgs'
copy_to = ['../../mpic_data/imgs_fromB_maj', '../../mpic_data/imgs_fromB_maj_verified',  '../../mpic_data/imgs_fromB_majmin_verified']

classes = ['aggregate', 'bubble', 'fiber_blur', 'fiber_sharp', 'long_pellet', 'mini_pellet', 'noise', 'phyto_dino', 'phyto_long', 'phyto_round', 'rhizaria', 'salp_pellet', 'short_pellet', 'swimmer']
for c in classes:
    filenames = [f for f in os.listdir(f'{copy_from}/{c}') if (f != '.DS_Store' and 'RR' not in f)]
    for f, d in product(filenames, copy_to):
        os.makedirs(f'{d}/{c}', exist_ok=True)
        shutil.copyfile(f'{copy_from}/{c}/{f}', f'{d}/{c}/{f}')

for dir in copy_to:  # testing if they all have the same # of images
    print(dir)
    all_filenames = []
    classes = [c for c in os.listdir(dir) if os.path.isdir(f'{dir}/{c}')]
    for c in classes:
        filenames = [f for f in os.listdir(f'{dir}/{c}') if f != '.DS_Store']
        all_filenames.extend(filenames)
    print(len(all_filenames))