import argparse

import yaml

import src.dataset as dataset
import src.tools as tools

parser = argparse.ArgumentParser()
parser.add_argument('--domain', '-d')
args = parser.parse_args()

model_name = f'target{args.domain}_ood'
cfg = yaml.safe_load(open(f'../configs/{model_name}.yaml', 'r'))

df = tools.load_metadata()
train_df = df.loc[df['label'] != 'none']

splits = {'train': [], 'val': []}
train_size = 0.8
for d in cfg['train_domains']:
    d_df = train_df.loc[train_df['domain'] == d]
    filepaths = dataset.stratified_split(d_df, train_size)
    splits['train'].extend(filepaths[0])
    splits['val'].extend(filepaths[1])
if cfg['target_domain'] is not None:
    test_df = df.loc[df['domain'] == cfg['target_domain']]
    test_fps = test_df['label'] + '/' + test_df['filename']
    splits['test']= list(test_fps.values)

tools.write_json(splits, f'../data/{model_name}.json')
