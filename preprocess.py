"""
Used for organizing and padding image files as they were received.

Here's a description of the directories involved in this preprocessing step:
- /labeled: contains subdirectories corresponding to original image labels as
annotated by Colleen. Images come from FK and RR cruises. Images that were too
small for flux calculations not included.
- /unlabeled: All JC images, including those that Colleen said to exclude
("For samples JC49 and above, exclude “_7x_collage” images). Organized by
original subdirectory names (with JC number IDs). Images that were too small
for flux calculations not included. 
- /combined: /unlabeled (with “_7x_collage” removed) and /labeled combined.
Has an index.csv file with filename and label columns. 
- /train: padded /combined images that are not from FK. Has index.csv with
object_id, label, and path columns. To be used for model training and testing.
- /eval: padded /combined images that are from FK. Has index.csv with
object_id, label, and path columns. To be used for out-of-domain model
evaluation.
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms.functional import rotate
    
def pad(image, square_size=128):
    """Rescale and center images along the longest axis, then zero-pad.

    Adapted from:
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

    Args:
        image (PIL.Image.Image): image to be padded

    Returns:
        padded_image (PIL.Image.Image): padded image
    """
    if image.size[0] > image.size[1]:  # if W > H, rotate so that H > W
        image = rotate(image, 90, expand=True)

    height = image.size[1]
    if height > square_size:  # rescale if image is larger than square
        ratio = square_size / height
        scaled_size = [int(x * ratio) for x in image.size]
        image = image.resize(scaled_size)
    else:
        scaled_size = image.size
    
    padded_image = Image.new('RGB', (square_size, square_size))
    paste_at = [(square_size - s) // 2 for s in scaled_size]
    padded_image.paste(image, paste_at)

    return padded_image


def write_index(path, columns, contents):
    """Create an index (csv) for image objects within a directory.

    Args:
        path (str): the path to the directory
        columns (tuple[str]): header column names
        contents (tuple[list-like]): contents corresponding to each of the
        columns
    """
    with open(os.path.join(path, 'index.csv'), 'w') as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(zip(*contents))


def make_dir(path):
    """Create a new directory at the specified path."""
    if not os.path.exists(path):
        os.makedirs(path)


def copy_imgs(copy_from, copy_to):
    """Copy images from one directory to another.

    Args:
        copy_from (str): path of the directory to copy from. The basename of
        the directory should be either /labeled or /unlabeled.
        copy_to (str): path of the directory to copy to

    Returns:
        filenames (list): names of image files that were copied
        labels (list): the corresponding labels of the files that were copied.
        Unlabeled images have corresponding entries equal to 'none'.
    """
    label_status = os.path.basename(copy_from)
    ids = []
    labels = []

    for path, _, files in os.walk(copy_from):
        for f in files:
            if label_status == 'labeled':
                label = os.path.basename(path)
                condition = not f.endswith('.jpg')
            else:
                label = 'none'
                condition = not f.endswith('.tiff') or '_7x_collage' in f
            if condition:
                continue
            object_id = f.split('.')[0]
            f_jpg = f'{object_id}.jpg'
            image = Image.open(os.path.join(path, f))
            image.save(os.path.join(copy_to, f_jpg), 'JPEG', quality=95)
            ids.append(object_id)
            labels.append(label)

    return ids, labels


def make_combined_dir(parent):
    """Combine the images from /unlabeled and /labeled directries."""
    unlabeled_path = os.path.join(parent, 'unlabeled')
    labeled_path = os.path.join(parent, 'labeled')
    combined_path = os.path.join(parent, 'combined')

    make_dir(combined_path)

    labeled_ids, labeled_labels = copy_imgs(labeled_path, combined_path)
    unlabeled_ids, unlabeled_labels = copy_imgs(unlabeled_path, combined_path)

    ids = labeled_ids + unlabeled_ids
    labels = labeled_labels + unlabeled_labels
    paths = [os.path.join('combined', f'{i}.jpg') for i in ids]

    columns = ('object_id', 'label', 'path')
    contents = (ids, labels, paths)
    write_index(combined_path, columns, contents)


def pad_combined_images(parent, square_size=128):
    """Pad images from /combined and separate them into train and eval sets."""
    combined_path = os.path.join(parent, 'combined')
    train_path = os.path.join(parent, 'train')

    make_dir(train_path)

    index = pd.read_csv(os.path.join(combined_path, 'index.csv'))
    train_ids = [i for i in index['object_id'].to_list() if 'FK' not in i]
    train_filepaths = [os.path.join('train', f'{i}.jpg') for i in train_ids]

    for i in train_ids:
        f = f'{i}.jpg'
        image = Image.open(os.path.join(combined_path, f))
        padded = pad(image, square_size)
        padded.save(os.path.join(train_path, f), quality=95)

    columns = ('object_id', 'path')
    train_contents = (train_ids, train_filepaths)

    write_index(train_path, columns, train_contents)


if __name__ == '__main__':

    path = '/Users/particle/imgs'
    # make_combined_dir(path)
    pad_combined_images(path)
