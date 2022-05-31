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
import os
import csv

import pandas as pd
from PIL import Image


def pad(im):
    """Rescale and center images along the longest axis, then zero-pad.

    Adapted from:
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

    Args:
        im (PIL.Image.Image): image to be padded

    Returns:
        padded_im (PIL.Image.Image): padded image
    """
    square_size = 224
    orig_size = im.size
    ratio = square_size / max(orig_size)
    scaled_size = [int(x * ratio) for x in orig_size]
    im = im.resize(scaled_size, Image.LANCZOS)
    padded_im = Image.new('RGB', (square_size, square_size))
    paste_at = [(square_size - s) // 2 for s in scaled_size]
    padded_im.paste(im, paste_at)

    return padded_im


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
    filenames = []
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
            im = Image.open(os.path.join(path, f))
            im.save(os.path.join(copy_to, f))
            filenames.append(f)
            labels.append(label)

    return filenames, labels


def make_combined_dir(parent):
    """Combine the images from /unlabeled and /labeled directries."""
    unlabeled_path = os.path.join(parent, 'unlabeled')
    labeled_path = os.path.join(parent, 'labeled')
    combined_path = os.path.join(parent, 'combined')

    make_dir(combined_path)

    l_filenames, l_labels = copy_imgs(labeled_path, combined_path)
    u_filenames, u_labels = copy_imgs(unlabeled_path, combined_path)

    filenames = l_filenames + u_filenames
    labels = l_labels + u_labels

    columns = ('filename', 'label')
    contents = (filenames, labels)
    write_index(combined_path, columns, contents)


def pad_combined_images(parent):
    """Pad images from /combined and separate them into train and eval sets."""
    def save_image(path, filename, ids, labels, filepaths):

            padded.save(os.path.join(path, filename))
            ids.append(f.split('.')[0])
            labels.append(l)
            filepaths.append(os.path.join(os.path.basename(path), filename))
        
    combined_path = os.path.join(parent, 'combined')
    train_path = os.path.join(parent, 'train')
    eval_path = os.path.join(parent, 'eval')

    make_dir(train_path)
    make_dir(eval_path)

    index = pd.read_csv(os.path.join(combined_path, 'index.csv'))
    all_filenames = index['filename'].to_list()
    all_labels = index['label'].to_list()
    train_ids = []
    train_labels = []
    train_filepaths = []
    eval_ids = []
    eval_labels = []
    eval_filepaths = []

    for f, l in zip(all_filenames, all_labels):
        image = Image.open(os.path.join(combined_path, f))
        padded = pad(image)
        if 'FK' in f:
            save_image(eval_path, f, eval_ids, eval_labels, eval_filepaths)
        else:
            save_image(train_path, f, train_ids, train_labels, train_filepaths)

    columns = ('object_id', 'label', 'path')
    train_contents = (train_ids, train_labels, train_filepaths)
    eval_contents = (eval_ids, eval_labels, eval_filepaths)

    write_index(train_path, columns, train_contents)
    write_index(eval_path, columns, eval_contents)


# def get_background_color(im):
#     """Get mean RGB values of an image"""
#     x, y = im.size

#     upper_left = im.getpixel((0, 0))
#     upper_right = im.getpixel((x - 1, 0))
#     lower_left = im.getpixel((0, y - 1))
#     lower_right = im.getpixel((x - 1, y - 1))
#     corners = (upper_left, upper_right, lower_left, lower_right)

#     mean_r = int(np.mean([c[0]**2 for c in corners])**0.5)
#     mean_g = int(np.mean([c[1]**2 for c in corners])**0.5)
#     mean_b = int(np.mean([c[2]**2 for c in corners])**0.5)

#     return (mean_r, mean_g, mean_b)


if __name__ == '__main__':

    path = '/Users/particle/imgs'
    pad_combined_images(path)
