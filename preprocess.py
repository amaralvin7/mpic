"""
Used for organizing and preprocessing image files as they were received.

Here's a description of the directories involved in this preprocessing step:
- /labeled: contains subdirectories corresponding to original image labels as
annotated by Colleen. Images come from FK and RR cruises. Images that were too
small for flux calculations not included.
- /unlabeled: All JC images, including those that Colleen said to exclude
("For samples JC49 and above, exclude “_7x_collage” images). Organized by
original subdirectory names (with JC number IDs). Images that were too small
for flux calculations not included. 
- /labeled_grouped: zooplankton and zooplankton_part were grouped.
"""
import os
import pandas as pd
import random
import shutil

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import rotate, resize


def make_dir(path):
    """Create a new directory at the specified path."""
    if not os.path.exists(path):
        os.makedirs(path)


def site_split(path, split_from, split_to, binary=False):

    lg_path = os.path.join(path, split_from)
    split_path = os.path.join(path, split_to)
    labels = [l for l in os.listdir(lg_path) if os.path.isdir(os.path.join(lg_path, l))]
    for l in labels:
        filenames = [f for f in os.listdir(os.path.join(lg_path, l)) if '.jpg' in f]
        if binary and l != 'unidentifiable':
            new_label = 'object'
        else:
            new_label = l
        make_dir(os.path.join(split_path, 'RR', new_label))
        make_dir(os.path.join(split_path, 'FK', new_label))
        for f in filenames:
            prefix = f[:2]
            shutil.copy(os.path.join(lg_path, l, f),
                        os.path.join(split_path, prefix, new_label, f))


def tv_split(path, split_from, val_size=0.2, subsample=None):

    grouped_path = os.path.join(path, split_from)
    split_path = os.path.join(path, f'{split_from}_tvsplit')
    make_dir(split_path)
    labels = [l for l in os.listdir(grouped_path) if os.path.isdir(os.path.join(grouped_path, l))]
    for l in labels:
        grouped_filenames = [f for f in os.listdir(os.path.join(grouped_path, l)) if '.jpg' in f]
        if subsample:
            grouped_filenames = random.sample(grouped_filenames, subsample)
        train_filenames, val_filenames = train_test_split(grouped_filenames, test_size=val_size, random_state=0)
        for f in train_filenames:
            make_dir(os.path.join(split_path, 'train', l))
            shutil.copy(os.path.join(grouped_path, l, f),
                        os.path.join(split_path, 'train', l, f))
        for f in val_filenames:
            make_dir(os.path.join(split_path, 'val', l))
            shutil.copy(os.path.join(grouped_path, l, f),
                        os.path.join(split_path, 'val', l, f))

           
def tvt_split(path, split_from, test_size=0.1):

    grouped_path = os.path.join(path, split_from)
    split_path = os.path.join(path, f'{split_from}_tvtsplit')
    make_dir(split_path)
    labels = [l for l in os.listdir(grouped_path) if os.path.isdir(os.path.join(grouped_path, l))]
    for l in labels:
        grouped_filenames = [f for f in os.listdir(os.path.join(grouped_path, l)) if '.jpg' in f]
        trainval_filenames, test_filenames = train_test_split(grouped_filenames, test_size=test_size, random_state=0)
        new_test_size = test_size/(1 - test_size)
        train_filenames, val_filenames = train_test_split(trainval_filenames, test_size=new_test_size, random_state=0)
        for f in train_filenames:
            make_dir(os.path.join(split_path, 'train', l))
            shutil.copy(os.path.join(grouped_path, l, f),
                        os.path.join(split_path, 'train', l, f))
        for f in val_filenames:
            make_dir(os.path.join(split_path, 'val', l))
            shutil.copy(os.path.join(grouped_path, l, f),
                        os.path.join(split_path, 'val', l, f))
        for f in test_filenames:
            make_dir(os.path.join(split_path, 'test', l))
            shutil.copy(os.path.join(grouped_path, l, f),
                        os.path.join(split_path, 'test', l, f))


def pad(image, square_size=128):
    """Rescale and center images along the longest axis, then zero-pad.

    Adapted from:
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

    Args:
        image (PIL.Image.Image): image to be padded

    Returns:
        padded_image (PIL.Image.Image): padded image
    """
    # if image.size[0] > image.size[1]:  # if W > H, rotate so that H > W
    #     image = rotate(image, 90, expand=True)

    max_dim = max(image.size)
    if max_dim > square_size:  # rescale if image is larger than square
        ratio = square_size / max_dim
        scaled_size = [int(x * ratio) for x in image.size]
        image = image.resize(scaled_size)
    else:
        scaled_size = image.size
    
    padded_image = Image.new('RGB', (square_size, square_size))
    paste_at = [(square_size - s) // 2 for s in scaled_size]
    padded_image.paste(image, paste_at)

    return padded_image


def pad_split_images(pad_from, tuple_of_dirs):
    
    make_dir(f'{pad_from}_pad')
    
    for phase in tuple_of_dirs:
        make_dir(os.path.join(f'{pad_from}_pad', phase))
        phase_path = os.path.join(pad_from, phase)
        labels = [l for l in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, l))]
        for l in labels:
            filenames = [f for f in os.listdir(os.path.join(phase_path, l)) if '.jpg' in f]
            for f in filenames:
                make_dir(os.path.join(f'{pad_from}_pad', phase, l))
                image = Image.open(os.path.join(pad_from, phase, l, f))
                rescaled = pad(image)
                rescaled.save(os.path.join(f'{pad_from}_pad', phase, l, f), quality=95) 


def pad_test_images(pad_from):
    
    make_dir(f'{pad_from}_pad')
    
    labels = [l for l in os.listdir(pad_from) if os.path.isdir(os.path.join(pad_from, l))]
    for l in labels:
        filenames = [f for f in os.listdir(os.path.join(pad_from, l)) if '.jpg' in f]
        for f in filenames:
            make_dir(os.path.join(f'{pad_from}_pad', l))
            image = Image.open(os.path.join(pad_from, l, f))
            rescaled = pad(image)
            rescaled.save(os.path.join(f'{pad_from}_pad', l, f), quality=95)


def pad_combined_images(pad_from):
    
    make_dir(f'{pad_from}_pad')
    
    filenames = [f for f in os.listdir(pad_from) if '.jpg' in f]
    for f in filenames:
        image = Image.open(os.path.join(pad_from, f))
        rescaled = pad(image)
        rescaled.save(os.path.join(f'{pad_from}_pad', f), quality=95) 
            

def count_samples(path):

    contents = os.listdir(path)
    labels = [name for name in contents if os.path.isdir(os.path.join(path, name))]
    labels.sort()
    count_dict = {}
    
    for l in labels:
        n_smps = len([f for f in os.listdir(os.path.join(path, l)) if f.endswith('.jpg')])
        count_dict[l] = n_smps
    
    plt.bar(range(len(count_dict)), list(count_dict.values()), align='center')
    plt.xticks(range(len(count_dict)), list(count_dict.keys()), rotation=90)
    plt.grid(axis='y', zorder=1)
    plt.tight_layout()
    plt.savefig('sample_counts')
    plt.close()


def combine_RR_FK(combine_from, combine_into):

    labels = [l for l in os.listdir(combine_from) if os.path.isdir(os.path.join(combine_from, l))]
    for l in labels:
        filenames = [f for f in os.listdir(os.path.join(combine_from, l)) if '.jpg' in f]
        for f in filenames:
            shutil.copy(os.path.join(combine_from, l, f),
                        os.path.join(combine_into, f))
            

def copy_verified_images():
    
    full_df = pd.read_csv('/Users/particle/Drive/ucsc/code/pico/Compared_classification.csv')
    copy_from = '/Users/particle/imgs/labeled'
    copy_to = '/Users/particle/imgs/relabel/verified'
    verified = full_df[full_df['CD_label1'] == full_df['CD_label2']].copy()
    verified = verified[verified['CD_label1'] != 'unidentifiable']
    verified_files = verified['file'].values
    verified_labels = verified['CD_label1'].values
    
    labels = [l for l in os.listdir(copy_from) if os.path.isdir(os.path.join(copy_from, l))]
    for l in labels:
        make_dir(os.path.join(copy_to, l))
        
    for f, l in zip(verified_files, verified_labels):
        shutil.copy(os.path.join(copy_from, l, f),
                    os.path.join(copy_to, l, f))

def make_combined_dir(parent):
    """Combine the images from /labeled directory."""
    labeled_path = os.path.join(parent, 'labeled')
    combined_path = os.path.join(parent, 'combined')

    make_dir(combined_path)

    for path, _, files in os.walk(labeled_path):
        for f in files:
            if f.endswith('.jpg'):
                shutil.copy(os.path.join(path, f),
                            os.path.join(combined_path, f))

    
if __name__ == '__main__':

    path = '/Users/particle/imgs'
    pad_test_images(os.path.join(path, 'relabel_20221014_sitesplit/FK'))

