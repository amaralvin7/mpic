import h5py
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
import os
import time
from itertools import product
from PIL import Image
from sklearn.preprocessing import StandardScaler

from preprocess import make_dir


def reduce_features(n_components):
    
    with h5py.File('features.h5', 'r') as f:
        features = np.array(f['features'])

    features_scaled = StandardScaler().fit_transform(features)

    features_umap = umap.UMAP(n_neighbors=30, min_dist=0, n_components=n_components,
                              random_state=0).fit_transform(features_scaled)

    with h5py.File('features.h5', 'a') as f:
        f.create_dataset('features_umap', data=features_umap, dtype='float32')


def get_clusterer(features, min_samples, method, score=False):

    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        cluster_selection_method=method,
        min_cluster_size=10,
        gen_min_span_tree=score
        ).fit(features)

    return clusterer


def param_search():

    with h5py.File('features.h5', 'r') as f:
        features_umap = np.array(f['features_umap'])
    
    min_samples = range(1, 11)
    methods = ['eom', 'leaf']
    combos = product(min_samples, methods)
    best_score = 0
    best_clusterer = None
    
    for min_samples, method in combos:

        clusterer = get_clusterer(features_umap, min_samples, method, True)
        clustered = (clusterer.labels_ >= 0)
        percent_clustered = np.sum(clustered) / features_umap.shape[0] * 100
        score = clusterer.relative_validity_
        
        print(f'Score (min_samples={min_samples}, method={method}): {score:.4f}')
        print(f'Percent clustered: {percent_clustered:.2f}')
        print('--------')

        if score > best_score:
            best_score = score
            best_clusterer = clusterer
            best_params = (min_samples, method)
            best_percent_clustered = percent_clustered
    
    print(f'Best params: min_samples={best_params[0]}, method={best_params[1]} ({best_percent_clustered:.2f}% objects clustered)')
    print(f'Number of labels: {len(set(best_clusterer.labels_))}')
    
    with h5py.File('features.h5', 'a') as f:
        f.create_dataset('labels', data=best_clusterer.labels_, dtype='int')


def images_by_label():

    with h5py.File('features.h5', 'r') as f:
        labels = list(f['labels'])
        ids = list(f['object_id'].asstr())
    
    for l in set(labels):
        make_dir(os.path.join('/Users/particle/imgs/umap', str(l)))
        
    for i, l in zip(ids, labels):
        image = Image.open(os.path.join('/Users/particle/imgs/combined', f'{i}.jpg'))
        image.save(os.path.join('/Users/particle/imgs/umap', str(l), f'{i}.jpg'), 'JPEG', quality=95)   


if __name__ == '__main__':
    
    start_time = time.time()
    
    reduce_features(50)
    param_search()
    images_by_label()
    
    print(f'--- {(time.time() - start_time)/60} minutes ---')