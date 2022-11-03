
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
import sys

from sklearn.metrics import ConfusionMatrixDisplay

from preprocess import make_dir

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


def confusion_matrix(df, col1, col2):

    agreement = sum(df[col1] == df[col2])
    print(f'Agreement percentage: {agreement/len(df)*100:.2f} (N = {agreement})')

    _, ax = plt.subplots(figsize=(10,10))
    ConfusionMatrixDisplay.from_predictions(
        y_true=df[col1],
        y_pred=df[col2],
        cmap=plt.cm.Blues,
        normalize=None,
        xticks_rotation='vertical',
        values_format='.0f',
        ax=ax
    )
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.tight_layout()
    plt.savefig(f'confusionmatrix')
    plt.close()


def copy_disagreements(df, col1, col2, col1_path):

    make_dir('disagreements')
    disagreements = df.loc[df[f'{col1}_r'] != df[f'{col2}_r']]
    for _, row in disagreements.iterrows():
        col1_label = row[col1]
        col2_label = row[col2]
        file = row['file']
        shutil.copy(os.path.join(col1_path, col1_label, file),
                    os.path.join('disagreements',
                                 f'{file.split(".")[0]}_{col1_label}_{col2_label}.jpg'))

def revise_old_labels(df, col):
    
    d = {'aggregate': 'aggregate', 'amphipod': 'swimmer', 'copepod': 'swimmer',
         'dense_detritus': 'aggregate', 'fiber': 'fiber',
         'foraminifera': 'swimmer', 'large_loose_pellet': 'long_pellet',
         'long_fecal_pellet': 'long_pellet', 'mini_pellet': 'mini_pellet',
         'phytoplankton': 'phytoplankton', 'pteropod': 'swimmer',
         'rhizaria': 'rhizaria', 'salp_pellet': 'salp_pellet',
         'short_pellet': 'short_pellet', 'unidentifiable': 'unidentifiable',
         'zooplankton': 'swimmer', 'zooplankton_part': 'swimmer'}
    
    df[f'{col}_r'] = df[col].apply(lambda x: d[x])

def revise_new_labels(df, col):
    
    d = {'aggregate': 'aggregate', 'bubble': 'unidentifiable',
         'fiber_blur': 'fiber', 'fiber_sharp': 'fiber',
         'long_pellet': 'long_pellet', 'mini_pellet': 'mini_pellet',
         'noise': 'unidentifiable', 'phyto_long': 'phytoplankton',
         'phyto_round': 'phytoplankton', 'rhizaria': 'rhizaria',
         'salp_pellet': 'salp_pellet', 'short_pellet': 'short_pellet',
         'skip': 'unidentifiable', 'swimmer': 'swimmer'}
    
    df[f'{col}_r'] = df[col].apply(lambda x: d[x])


# CD_labels = labels_to_df('/Users/particle/imgs/labeled', 'CD')

CD_labels = pd.read_csv('Compared_classification.csv')
CD_labels = CD_labels.drop('CD_label1', axis=1)
CD_labels.rename(columns={'CD_label2': 'CD_label'}, inplace=True)

revise_old_labels(CD_labels, 'CD_label')
predictions = pd.read_csv('predictions.csv')
revise_new_labels(predictions, 'predicted_label')
df = CD_labels.merge(predictions, on='file')
print(len(df))
print(df.head())
df.to_csv('label_comparison.csv', index=False)
confusion_matrix(df, 'CD_label_r', 'predicted_label_r')

col1 = 'CD_label'
col2 = 'predicted_label'
col1_path = '/Users/particle/imgs/labeled'
copy_disagreements(df, col1, col2, col1_path)