import os
import shutil
import sys

import pandas as pd
import yaml

import src.dataset as dataset


def create_prediction_dir(prediction_dir):

    prediction_path = f'../../mpic_data/{prediction_dir}'

    if os.path.exists(prediction_path) and os.path.isdir(prediction_path):  # delete if exists, otherwise create it
        shutil.rmtree(prediction_path)
    else:
        os.makedirs(prediction_path)

    return prediction_path


def copy_class_imgs(c, filepath_list, copy_to):

    os.makedirs(f'{copy_to}/{c}')
    for i in filepath_list:
        old_filepath = f'{copy_from}/{i}'
        new_filepath = f'{copy_to}/{c}/{i.split("/")[1]}'
        shutil.copyfile(old_filepath, new_filepath)



replicates = 5
topN = 1000
copy_from = '../../mpic_data/imgs'
cfg = yaml.safe_load(open('../configs/hitloopI/B.yaml', 'r'))
classes = cfg['classes']
train_filepaths = dataset.compile_filepaths(cfg, cfg['train_domains'], 'train')

# COPY IMAGES USING AN ENSEMBLE VOTING APPROACH
path_maj = create_prediction_dir('imgs_fromB_maj')
path_min = create_prediction_dir('imgs_fromB_min')

df_list = []
for i in range(replicates):
    temp_df = pd.read_csv(f'../results/hitloopI/predictions/B-{i}.csv', index_col='filepath', header=0)
    temp_df = temp_df.rename(columns={c: f'{c}{i}' for c in temp_df.columns})
    df_list.append(temp_df)
df = pd.concat(df_list, axis=1)

df_rows = []
prediction_cols = [c for c in df.columns if 'prediction' in c]
for i, r in df.iterrows():
    vote = max(set(r[prediction_cols]), key=list(r[prediction_cols].values).count)
    n_vote = sum(r[prediction_cols] == vote)
    if n_vote == replicates:  # ~51% of predictions
        vote_cols = [c for c in r.index if vote in c]
        df_rows.append({'filepath': i, 'prediction': vote, 'ensemble_mean': r[vote_cols].mean()})
df_voted = pd.DataFrame(df_rows)
df_voted.set_index('filepath', drop=True, inplace=True)

for c in classes:
    class_fps = [f for f in train_filepaths if c in f]
    if len(class_fps) >= 100:
        top_predictions = df_voted.loc[df_voted['prediction'] == c].nlargest(topN, 'ensemble_mean', keep='all')
        copy_class_imgs(c, top_predictions.index.values, path_maj)
    else:
        topN_by_replicate = []
        for i in range(replicates):
            top_predictions = df.nlargest(topN, f'{c}{i}', keep='all')
            topN_by_replicate.append(top_predictions.index.values)
        topN_intersection = list(set.intersection(*map(set, topN_by_replicate)))
        copy_class_imgs(c, topN_intersection, path_min)
