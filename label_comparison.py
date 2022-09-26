
import matplotlib.pyplot as plt
import os
import pandas as pd
import shutil
import sys

from sklearn.metrics import ConfusionMatrixDisplay

from preprocess import make_dir

df = pd.read_csv('Compared_classification.csv')
df['CD_label2_grouped'] = df['CD_label2']

# CD's labels to be grouped
swimmer = ('zooplankton', 'zooplankton_part', 'copepod', 'amphipod', 'pteropod', 'foraminifera')
long_pellet = ('long_fecal_pellet', 'large_loose_pellet')

# VA's groups to be grouped
unidentifiable = ('skip', 'noise', 'bubble')
fiber = ('fiber_sharp', 'fiber_blur')
phytoplankton = ('phyto_round', 'phyto_long')

df.loc[df['CD_label2'].isin(swimmer), 'CD_label2_grouped'] = 'swimmer'
df.loc[df['CD_label2'] == 'dense_detritus', 'CD_label2_grouped'] = 'aggregate'
df.loc[df['CD_label2'].isin(long_pellet), 'CD_label2_grouped'] = 'long_pellet'

# put VA's labels into a df, and merge them with CD's
relabel_path = '/Users/particle/imgs/relabel/finish'
labels = [l for l in os.listdir(relabel_path) if os.path.isdir(os.path.join(relabel_path, l))]
list_of_labels = []
list_of_filenames = []
for l in labels:
    filenames = [f for f in os.listdir(os.path.join(relabel_path, l)) if '.jpg' in f]
    list_of_filenames.extend(filenames)
    list_of_labels.extend([l] * len(filenames))
df2 = pd.DataFrame(list(zip(list_of_filenames, list_of_labels, list_of_labels)), columns =['file', 'VA_label', 'VA_label_grouped'])
df2.loc[df2['VA_label_grouped'].isin(unidentifiable), 'VA_label_grouped'] = 'unidentifiable'
df2.loc[df2['VA_label_grouped'].isin(fiber), 'VA_label_grouped'] = 'fiber'
df2.loc[df2['VA_label_grouped'].isin(phytoplankton), 'VA_label_grouped'] = 'phytoplankton'
df2 = df2[df2['VA_label_grouped'] != 'unidentifiable']  # drop VA's unidentifiables
df = df.merge(df2, on='file')

#calculate agreement percentage between CD and VA, and calculate the confusion matrix
agreement = sum(df['CD_label2_grouped'] == df['VA_label_grouped'])
print(f"Number of VA's identifiable images labeled by CD: {len(df)}")
print(f'Agreement percentage: {agreement/len(df)*100:.0f} (N = {agreement})')
fig, ax = plt.subplots(figsize=(10,10))
disp = ConfusionMatrixDisplay.from_predictions(
    y_true=df['CD_label2_grouped'],
    y_pred=df['VA_label_grouped'],
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
disagreements = df.loc[df['CD_label2_grouped'] != df['VA_label_grouped']]
for _, row in disagreements.iterrows():
    cd_label = row['CD_label2_grouped']
    va_label_grouped = row['VA_label_grouped']
    va_label_ungrouped = row['VA_label']
    file = row['file']
    shutil.copy(os.path.join(relabel_path, va_label_ungrouped, file),
                os.path.join('disagreements', f'{file.split(".")[0]}_CD_{cd_label}_VA_{va_label_ungrouped}.jpg'))