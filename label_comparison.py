
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

# put VA's labels into a df, and merge them with CD's
va_path = '/Users/particle/imgs/relabel_20220926'
df = labels_to_df(va_path, 'VA')
df2 = labels_to_df('/Users/particle/imgs/relabel_20220920_CD', 'CD')

df = df.merge(df2, on='file')

#calculate agreement percentage between CD and VA, and calculate the confusion matrix
agreement = sum(df['CD_label'] == df['VA_label'])
print(f'Agreement percentage: {agreement/len(df)*100:.2f} (N = {agreement})')

fig, ax = plt.subplots(figsize=(10,10))
disp = ConfusionMatrixDisplay.from_predictions(
    y_true=df['CD_label'],
    y_pred=df['VA_label'],
    # display_labels=test_data.classes,
    cmap=plt.cm.Blues,
    normalize=None,
    xticks_rotation='vertical',
    values_format='.0f',
    ax=ax
)
plt.xlabel('Vinicius')
plt.ylabel('Colleen')
plt.tight_layout()
plt.savefig('CD_VA_confusionmatrix')
plt.close()

# copy disagreements into a folder
make_dir('disagreements')
disagreements = df.loc[df['CD_label'] != df['VA_label']]
for _, row in disagreements.iterrows():
    cd_label = row['CD_label']
    va_label = row['VA_label']
    file = row['file']
    shutil.copy(os.path.join(va_path, va_label, file),
                os.path.join('disagreements', f'{file.split(".")[0]}_CD_{cd_label}_VA_{va_label}.jpg'))