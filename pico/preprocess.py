# https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
from PIL import Image, ImageOps
import numpy as np
import os

def pad(im):
    
    square_size = 224
    orig_size = im.size
    ratio = square_size / max(orig_size)
    scaled_size = [int(x * ratio) for x in orig_size]
    im = im.resize(scaled_size, Image.LANCZOS)
    new_im = Image.new('RGB', (square_size, square_size))
    paste_at = [(square_size - s) // 2 for s in scaled_size]
    new_im.paste(im, paste_at)
    
    return new_im

def pad_examples(path):

    copepod = f'{path}orig/copepod/FK170124_NBST1_7x_obl_1_3656.jpg'
    fiber = f'{path}orig/fiber/FK170124_NBST3_20x_obl_4a_21954.jpg'
    pteropod = f'{path}orig/pteropod/RR1_7x_obl_11_127.jpg'
    pellet = f'{path}orig/short_pellet/FK170124_NBST1_50x_obl_17g_2151.jpg'
 
    for image in (copepod, fiber, pteropod, pellet):   
        orig = Image.open(image)
        padded = pad(orig)
        padded.show()

def pad_all_images(basepath):
    
    for path, _, files in os.walk(f'{basepath}orig'):
        for filename in files:
            if '.jpg' in filename:
                label = os.path.basename(path)
                image = Image.open(f'{path}/{filename}')
                padded = pad(image)
                fn_no_xt = filename.split('.')[0]
                # padded.save(f'{basepath}padded/{filename}.jpg')
                padded.save(f'{basepath}padded/{fn_no_xt}_{label}.jpg')

# def get_background_color(im):
    
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
    
    basepath = '/Users/particle/imgs/'
    pad_all_images(basepath)

