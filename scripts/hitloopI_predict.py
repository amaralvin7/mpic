import os
import yaml

import src.dataset as dataset
import src.predict as predict
import src.tools as tools


def label_plus_filename(df):

    fpaths = df['label'] + '/' + df['filename']

    return fpaths.values

exp_name = 'hitloopI'
target_domain = 'RR'

metadata = tools.load_metadata()

B_df = metadata.loc[metadata['domain'] == target_domain]  # target is all RR images
B_fpaths = label_plus_filename(B_df)
       
A_df = B_df.loc[B_df['label'] == 'none']  # target is only ambiguous RR images
A_fpaths = label_plus_filename(A_df)

models = [m for m in os.listdir(f'../results/{exp_name}/weights/') if '.pt' in m]

for m in models:

    model_id = m.split('.')[0]
    model_name = model_id.split('-')[0]

    if model_name == 'A':
        fps = A_fpaths
    else:
        fps = B_fpaths
    cfg = yaml.safe_load(open(f'../configs/{exp_name}/{model_name}.yaml', 'r'))
    
    loader = dataset.get_dataloader(cfg, fps, train=False, shuffle=False)
    predict.predict_labels(cfg['device'], loader, exp_name, model_id)
    