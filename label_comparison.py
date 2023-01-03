
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn import metrics


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


def revise_model_labels(df, columns):
    
    d = {'aggregate': 'aggregate', 'bubble': 'unidentifiable',
         'fiber_blur': 'fiber', 'fiber_sharp': 'fiber',
         'long_pellet': 'long_pellet', 'mini_pellet': 'mini_pellet',
         'noise': 'unidentifiable', 'phyto_long': 'phytoplankton',
         'phyto_round': 'phytoplankton', 'rhizaria': 'rhizaria',
         'salp_pellet': 'salp_pellet', 'short_pellet': 'short_pellet',
         'skip': 'unidentifiable', 'swimmer': 'swimmer'}
    
    for c in columns:
        df[f'{c}_r'] = df[c].apply(lambda x: d[x])


def compare_CD_agreement():

    # group CD's labels into revised categories and compare her two attempts
    CD_labels = pd.read_csv('labeling_attempts.csv')
    revise_CD_labels(CD_labels, ('CD_1', 'CD_2'))
    print(f'CD aggreement rate: {sum(CD_labels["CD_1_r"] == CD_labels["CD_2_r"]) / len(CD_labels) * 100:.0f}%')
    

def compare_JC_labels():

    CD_labels = pd.read_csv('JC_manually_classified_subset.csv')
    revise_CD_labels(CD_labels, ('ID',))
      
    model_labels = pd.read_csv('JC_predictions.csv')
    agreement_rates = []
    columns = [c for c in model_labels.columns if 'prediction' in c]
    revise_model_labels(model_labels, columns)
    
    for c in columns:

        df = model_labels[['file', f'{c}_r']]
        replicate = int(c.split('_')[1])

        merged = CD_labels.merge(df)
        agreement = sum(merged["ID_r"] == merged[f"{c}_r"]) / len(CD_labels) * 100
        agreement_rates.append(agreement)
        print(f'Aggreement rate, replicate {replicate}: {agreement:.0f}%')

        _, ax = plt.subplots(figsize=(10, 10))
        metrics.ConfusionMatrixDisplay.from_predictions(
            merged['ID_r'],
            merged[f'{c}_r'],
            cmap=plt.cm.Blues,
            normalize=None,
            xticks_rotation='vertical',
            values_format='.0f',
            ax=ax)
        ax.set_title('Confusion matrix')
        ax.set_ylabel('Human')
        ax.set_xlabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(f'JC_confusionmatrix_{replicate}'))
        plt.close()

    print(f'Overall agreement rate: {np.mean(agreement_rates):.2f} ± {np.std(agreement_rates, ddof=1):.2f}')

if __name__ == '__main__':
    
    compare_JC_labels()
