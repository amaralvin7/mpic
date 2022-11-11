import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from PIL import Image

def plot_histograms(path, exts, cutoff):
    '''Plot histograms of the largest dimensions of images'''
    def make_plot(paths, title):
        
        long_dims = np.zeros(len(paths))
        for i, p in enumerate(paths):
            with Image.open(p) as image:
                long_dims[i] = max(image.size)
    
        long_cutoff = long_dims[long_dims <= cutoff]
    
        _, ax = plt.subplots(tight_layout=True)
        ax.hist(long_cutoff, bins=100)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{title} % below {cutoff}: {len(long_cutoff)/len(long_dims)*100:.0f}')
        ax.set_xlabel('Length (pixels)')
        
        plt.savefig(f'{title}_pixels')
        plt.close()

    labeled_dir = os.path.join(path, 'relabel_20221110')
    labels = [l for l in os.listdir(labeled_dir) if os.path.isdir(os.path.join(labeled_dir, l))]
    l_paths = []
    for l in labels:
        l_paths.extend([os.path.join(labeled_dir, l, f) for f in os.listdir(os.path.join(labeled_dir, l)) if f.split('.')[1] in exts])
    rr_paths = [p for p in l_paths if 'RR' in p]
    fk_paths = [p for p in l_paths if 'FK' in p]
    srt_paths = [p for p in l_paths if 'SRT' in p]
    
    unlabeled_dir = os.path.join(path, 'unlabeled')
    jc_paths = [os.path.join(unlabeled_dir, f) for f in os.listdir(unlabeled_dir) if f.split('.')[1] in exts]
    
    all_paths = l_paths + jc_paths

    make_plot(all_paths, 'ALL')
    make_plot(jc_paths, 'JC')
    make_plot(rr_paths, 'RR')
    make_plot(fk_paths, 'FK')
    make_plot(srt_paths, 'SRT')

if __name__ =='__main__':
    
    exts = ('jpg', 'tiff')
    path = '/Users/particle/imgs'
    cutoff = 2224
    plot_histograms(path, exts, cutoff)