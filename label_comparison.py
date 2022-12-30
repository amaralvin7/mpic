
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
import sys


def labels_to_df(path, prefix):

    labels = [l for l in os.listdir(path) if os.path.isdir(os.path.join(path, l))]
    list_of_labels = []
    list_of_filenames = []
    for l in labels:
        filenames = [f for f in os.listdir(os.path.join(path, l)) if '.jpg' in f]
        list_of_filenames.extend(filenames)
        list_of_labels.extend([l] * len(filenames))
    df = pd.DataFrame(list(zip(list_of_filenames, list_of_labels)), columns =['file', f'{prefix}_label'])
    
    return df


def revise_CD_labels(df, columns):
    
    d = {'aggregate': 'aggregate', 'amphipod': 'swimmer', 'copepod': 'swimmer',
         'dense_detritus': 'aggregate', 'fiber': 'fiber',
         'foraminifera': 'swimmer', 'large_loose_pellet': 'long_pellet',
         'long_fecal_pellet': 'long_pellet', 'mini_pellet': 'mini_pellet',
         'phytoplankton': 'phytoplankton', 'pteropod': 'swimmer',
         'rhizaria': 'rhizaria', 'salp_pellet': 'salp_pellet',
         'short_pellet': 'short_pellet', 'unidentifiable': 'unidentifiable',
         'zooplankton': 'swimmer', 'zooplankton_part': 'swimmer'}
    
    for c in columns:
        df[f'{c}_r'] = df[c].apply(lambda x: d[x])


def revise_model_labels(df, col):
    
    d = {'aggregate': 'aggregate', 'bubble': 'unidentifiable',
         'fiber_blur': 'fiber', 'fiber_sharp': 'fiber',
         'long_pellet': 'long_pellet', 'mini_pellet': 'mini_pellet',
         'noise': 'unidentifiable', 'phyto_long': 'phytoplankton',
         'phyto_round': 'phytoplankton', 'rhizaria': 'rhizaria',
         'salp_pellet': 'salp_pellet', 'short_pellet': 'short_pellet',
         'skip': 'unidentifiable', 'swimmer': 'swimmer'}
    
    df[f'{col}_r'] = df[col].apply(lambda x: d[x])


def compare_CD_agreement():

    # group CD's labels into revised categories and compare her two attempts
    CD_labels = pd.read_csv('labeling_attempts.csv')
    revise_CD_labels(CD_labels, ('CD_1', 'CD_2'))
    print(f'CD aggreement rate: {sum(CD_labels["CD_1_r"] == CD_labels["CD_2_r"]) / len(CD_labels) * 100:.0f}%')

if __name__ == '__main__':
    
    compare_CD_agreement()
