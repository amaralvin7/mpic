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
import random
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import rotate, resize

    
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


def make_dir(path):
    """Create a new directory at the specified path."""
    if not os.path.exists(path):
        os.makedirs(path)


def rescale(path, func):

    path = '/Users/particle/imgs/'
    funcname = func.__name__
    lg_path = os.path.join(path, 'labeled_grouped')
    lgf_path = os.path.join(path, f'labeled_grouped_{funcname}')
    labels = [l for l in os.listdir(lg_path) if os.path.isdir(os.path.join(lg_path, l))]
    for l in labels:
        make_dir(os.path.join(lgf_path, 'RR', l))
        make_dir(os.path.join(lgf_path, 'FK', l))
        filenames = [f for f in os.listdir(os.path.join(lg_path, l)) if '.jpg' in f]
        for f in filenames:
            prefix = f[:2]
            image = Image.open(os.path.join(lg_path, l, f))
            if funcname == 'resize':
                rescaled = func(image, (128, 128))
            else:
                rescaled = func(image)
            rescaled.save(os.path.join(lgf_path, prefix, l, f), quality=95)


def tt_split(path, split_dir, binary=False):

    lg_path = os.path.join(path, 'labeled_grouped')
    split_path = os.path.join(path, split_dir)
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

def tv_split(path, split_from, split_to, val_size=0.2, subsample=None):

    grouped_path = os.path.join(path, split_from)
    split_path = os.path.join(path, split_to)
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


def pad_tvsplit_images():
    
    path = '/Users/particle/imgs/labeled_grouped_ttsplit/RR_1p4k_tvsplit'
    make_dir(f'{path}_pad')
    
    for phase in ('train', 'val'):
        make_dir(os.path.join(f'{path}_pad', phase))
        phase_path = os.path.join(path, phase)
        labels = [l for l in os.listdir(phase_path) if os.path.isdir(os.path.join(phase_path, l))]
        for l in labels:
            filenames = [f for f in os.listdir(os.path.join(phase_path, l)) if '.jpg' in f]
            for f in filenames:
                make_dir(os.path.join(f'{path}_pad', phase, l))
                image = Image.open(os.path.join(path, phase, l, f))
                rescaled = pad(image)
                rescaled.save(os.path.join(f'{path}_pad', phase, l, f), quality=95)  
            

if __name__ == '__main__':

    path = '/Users/particle/imgs'
    # tt_split(path, 'labeled_grouped_ttsplit_binary', True)
    tv_split(path, 'labeled_grouped_ttsplit/RR_1p4k', 'labeled_grouped_ttsplit/RR_1p4k_tvsplit', subsample=1400)
    pad_tvsplit_images()
