import os
import shutil

import pandas as pd
import yaml


def copy_class_imgs(c, filepath_list, copy_to):

    os.makedirs(f'{copy_to}/{c}')
    for i in filepath_list:
        old_filepath = f'{copy_from}/{i}'
        new_filepath = f'{copy_to}/{c}/{i.split("/")[1]}'
        shutil.copyfile(old_filepath, new_filepath)


replicates = 5
topN = 1000
copy_from = '../../mpic_data/imgs'
cfg = yaml.safe_load(open('../configs/hitloopI/A.yaml', 'r'))
classes = cfg['classes']

copy_to = '../../mpic_data/imgs_minboost'

df_list = []
for i in range(replicates):
    temp_df = pd.read_csv(f'../results/hitloopI/predictions/A-{i}.csv', index_col='filepath', header=0)
    temp_df = temp_df.rename(columns={c: f'{c}{i}' for c in temp_df.columns})
    df_list.append(temp_df)
df = pd.concat(df_list, axis=1)

for c in classes:
    class_fps = [f for f in os.listdir(f'../../mpic_data/imgs_fromA_verified/{c}') if 'RR' in f]
    if len(class_fps) < 100:
        topN_by_replicate = []
        for i in range(replicates):
            top_predictions = df.nlargest(topN, f'{c}{i}', keep='all')
            topN_by_replicate.append(top_predictions.index.values)
        topN_intersection = list(set.intersection(*map(set, topN_by_replicate)))
        to_copy = [f for f in topN_intersection if f.split('/')[1] not in class_fps]
        copy_class_imgs(c, to_copy, copy_to)

