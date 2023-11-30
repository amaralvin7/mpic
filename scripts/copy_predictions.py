import argparse
import os
import shutil

import pandas as pd
import yaml

import src.predict as predict
import src.tools as tools


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_list_id', '-i')
args = parser.parse_args()
list_id = int(args.model_name_list_id)
model_name = tools.get_model_names(list_id)[0]

replicates = 5
topN = 1000
copy_from = '../../mpic_data/imgs'
cfg = yaml.safe_load(open(f'../configs/{model_name}.yaml', 'r'))
classes = cfg['classes']

# COPY IMAGES USING AN ENSEMBLE VOTING APPROACH
prediction_path = f'../../mpic_data/imgs_from{list_id}'

if os.path.exists(prediction_path) and os.path.isdir(prediction_path):  # delete if exists, otherwise create it
    shutil.rmtree(prediction_path)
else:
    os.makedirs(prediction_path)

df = predict.get_prediction_df(model_name)

df_rows = []
prediction_cols = [c for c in df.columns if 'prediction' in c]
for i, r in df.iterrows():
    vote = max(set(r[prediction_cols]), key=list(r[prediction_cols].values).count)
    n_vote = sum(r[prediction_cols] == vote)
    if n_vote == replicates:
        vote_cols = [c for c in r.index if vote in c]
        df_rows.append({'filepath': i, 'prediction': vote, 'ensemble_mean': r[vote_cols].mean()})
df_voted = pd.DataFrame(df_rows)
df_voted.set_index('filepath', drop=True, inplace=True)

for c in classes:
    top_predictions = df_voted.loc[df_voted['prediction'] == c].nlargest(topN, 'ensemble_mean', keep='all')
    tools.copy_class_imgs(c, top_predictions.index.values, copy_from, prediction_path)
