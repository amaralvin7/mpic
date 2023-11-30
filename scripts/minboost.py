import argparse
import os

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
target_domain = cfg['target_domain']

copy_to = f'../../mpic_data/imgs_from{list_id}_minboost'
os.makedirs(copy_to)

df = predict.get_prediction_df(model_name)

for c in classes:
    class_fps = [f for f in os.listdir(f'../../mpic_data/imgs_from{list_id}_verified/{c}') if target_domain in f]
    if len(class_fps) < 100:
        topN_by_replicate = []
        for i in range(replicates):
            top_predictions = df.nlargest(topN, f'{c}{i}', keep='all')
            topN_by_replicate.append(top_predictions.index.values)
        topN_intersection = list(set.intersection(*map(set, topN_by_replicate)))
        to_copy = [f for f in topN_intersection if f.split('/')[1] not in class_fps]
        tools.copy_class_imgs(c, to_copy, copy_from, copy_to)

