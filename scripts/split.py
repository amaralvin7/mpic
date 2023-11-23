import yaml

import src.dataset as dataset
import src.tools as tools

model_name = tools.get_model_names(1)
cfg = yaml.safe_load(open(f'../configs/{model_name}.yaml', 'r'))

df = tools.load_metadata()
df = df.loc[df['label'] != 'none']

splits = {'train': [], 'val': []}
train_size = 0.8
for d in cfg['train_domains']:
    d_df = df.loc[df['domain'] == d]
    filepaths = dataset.stratified_split(d_df, train_size)
    splits['train'].extend(filepaths[0])
    splits['val'].extend(filepaths[1])

tools.write_json(splits, f'../data/{model_name}.json')
