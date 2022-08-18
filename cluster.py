import h5py
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import umap
import hdbscan
import os
import time
from itertools import product
from PIL import Image
from sklearn.preprocessing import StandardScaler


def reduce_features(n_components=50):
    
    with h5py.File('features.h5', 'r') as f:
        features = np.array(f['features'])

    features_scaled = StandardScaler().fit_transform(features)

    features_umap = umap.UMAP(n_neighbors=20, min_dist=0, n_components=n_components,
                              random_state=0).fit_transform(features_scaled)

    with h5py.File('features_umap.h5', 'w') as f:
        f.create_dataset('features_umap', data=features_umap, dtype='float32')


def get_cluster(args):

    min_samples, min_cluster_size, features_umap, write_labels = args
    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        cluster_selection_method='leaf',
        core_dist_n_jobs=1,
        gen_min_span_tree=True
        ).fit(features_umap)
    clustered = (clusterer.labels_ >= 0)
    percent_clustered = np.sum(clustered) / features_umap.shape[0] * 100
    n_labels = len(set(clusterer.labels_))
    score = clusterer.relative_validity_
    
    # clusterer.condensed_tree_.plot()
    # plt.savefig(f'tree_{min_samples}_{min_cluster_size}.pdf')
    # plt.close()
    
    print(min_samples, min_cluster_size, f'{percent_clustered:.2f}', f'{score:.2f}', n_labels)

    if write_labels:
        with h5py.File('labels.h5', 'w') as f:
            f.create_dataset('labels', data=clusterer.labels_, dtype='int')
    

def param_search():

    features_umap = load_umap_features()
    min_samples = (1, 2, 5, 10, 20)
    min_cluster_size = (20, 50, 100, 200, 500)
    
    inputs = list(product(min_samples, min_cluster_size, [features_umap], [False]))
    with mp.Pool(len(inputs), maxtasksperchild=1) as p:
        p.imap_unordered(get_cluster, inputs)
        p.close()
        p.join()


def load_umap_features():

    with h5py.File('features_umap.h5', 'r') as f:
        features_umap = np.array(f['features_umap'])
        
    return features_umap

def images_by_label():

    from preprocess import make_dir

    with h5py.File('features.h5', 'r') as f:
        ids = list(f['object_id'].asstr())
    
    with h5py.File('labels.h5', 'r') as f:
        labels = list(f['labels'])
        
    for l in set(labels):
        make_dir(os.path.join('/Users/particle/imgs/umap', str(l)))
        
    for i, l in zip(ids, labels):
        image = Image.open(os.path.join('/Users/particle/imgs/combined', f'{i}.jpg'))
        image.save(os.path.join('/Users/particle/imgs/umap', str(l), f'{i}.jpg'), 'JPEG', quality=95)   


if __name__ == '__main__':
    
    start_time = time.time()
    
    # reduce_features()
    param_search()
    # get_cluster([5, 500, load_umap_features(), True])
    # images_by_label()
    
    print(f'--- {(time.time() - start_time)/60} minutes ---')