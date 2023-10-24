import os
import shutil

import pandas as pd


def create_prediction_dir(prediction_dir):

    prediction_path = f'../../mpic_data/{prediction_dir}'

    if os.path.exists(prediction_path) and os.path.isdir(prediction_path):  # delete if exists, otherwise create it
        shutil.rmtree(prediction_path)
    else:
        os.makedirs(prediction_path)

    return prediction_path


replicates = 5
model_name = 'B'
copy_from = '../../mpic_data/imgs'
classes = ['aggregate', 'bubble', 'fiber_blur', 'fiber_sharp', 'long_pellet', 'mini_pellet', 'noise', 'phyto_dino', 'phyto_long', 'phyto_round', 'rhizaria', 'salp_pellet', 'short_pellet', 'swimmer']

# COPY TOP 200 IMAGE PREDICTIONS FOR EACH REPLICATE AND FOR EACH CLASS
top_N = 200
prediction_path = create_prediction_dir('imgs_fromB')

for i in range(replicates):
    replicate_path = f'{prediction_path}/{i}'
    os.makedirs(replicate_path)
    df = pd.read_csv(f'../results/hitloopI/predictions/B-{i}.csv')
    for c in classes:
        top_predictions = df.loc[df['prediction'] == c].nlargest(top_N, c, keep='all')
        if len(top_predictions) > 1:  # can't train/test split otherwise
            class_path = f'{replicate_path}/{c}'
            os.makedirs(class_path)
            for filepath in top_predictions['filepath'].values:
                filename = filepath.split('/')[1]
                shutil.copyfile(f'{copy_from}/{filepath}', f'{class_path}/{filename}')


# COPY IMAGES USING AN ENSEMBLE VOTING APPROACH
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
    # if n_vote == replicates:  # ~51% of predictions
    if n_vote >= replicates - 1:  #~73% of predictions
        vote_cols = [c for c in r.index if vote in c]
        df_rows.append({'filepath': i, 'prediction': vote, 'ensemble_mean': r[vote_cols].mean()})
df_voted = pd.DataFrame(df_rows)
df_voted.set_index('filepath', drop=True, inplace=True)

for top_N in (200, 400):
    prediction_path = create_prediction_dir(f'imgs_fromB_voted{top_N}')
    for c in classes:
        top_predictions = df_voted.loc[df_voted['prediction'] == c].nlargest(top_N, 'ensemble_mean', keep='all')
        if len(top_predictions) > 1:  # can't train/test split otherwise
            os.makedirs(f'{prediction_path}/{c}')
            for i, r in top_predictions.iterrows():
                old_filepath = f'{copy_from}/{i}'
                new_filepath = f'{prediction_path}/{r["prediction"]}/{i.split("/")[1]}'
                shutil.copyfile(old_filepath, new_filepath)