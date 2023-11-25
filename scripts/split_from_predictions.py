import argparse
import os
import shutil
import sys
import pandas as pd
import yaml

import src.dataset as dataset
import src.tools as tools

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_list_id', '-i')
args = parser.parse_args()
list_id = int(args.model_name_list_id)
model_names = tools.get_model_names(list_id)
copy_from = '../../mpic_data/imgs'
cfgs = [yaml.safe_load(open(f'../configs/{m}.yaml', 'r')) for m in model_names]
metadata = tools.load_metadata()
copy_to_list = []

for i, cfg in enumerate(cfgs):

    copy_to = cfg["data_dir"]
    copy_to_list.append(copy_to)
    ood_splits = tools.load_json(f'../data/target{cfg["target_domain"]}_ood.json')
    splits_dict = {'train': ood_splits['train'] + ood_splits['val']}  #all OOD images in train set

    df_rows = []
    for c in cfg['classes']:
        os.makedirs(f'{copy_to}/{c}', exist_ok=True)
        target_filenames = [f for f in os.listdir(f'{copy_to}/{c}') if f != '.DS_Store']
        for f in target_filenames:  # target images to copy
            df_rows.append({'filename': f, 'label': c})
        ood_filenames = [f for f in os.listdir(f'{copy_from}/{c}') if (f != '.DS_Store' and cfg["target_domain"] not in f)]
        for f in ood_filenames:  # OOD images to copy
            shutil.copyfile(f'{copy_from}/{c}/{f}', f'{copy_to}/{c}/{f}')

    labeled_df = pd.DataFrame(df_rows)
    train_fps, val_fps = dataset.stratified_split(labeled_df, 0.8)  # stratify based on new labels
    meta_targ = metadata.loc[metadata['domain'] == cfg['target_domain']][['filename', 'label']]
    test_df = meta_targ.merge(labeled_df, on='filename', how='left', indicator=True)  # all RR images not in folder
    test_df = test_df.loc[test_df['_merge'] == 'left_only']
    test_df['filepath'] = test_df['label_x'] + '/' + test_df['filename']

    splits_dict['train'].extend(train_fps)
    splits_dict['val'] = val_fps
    splits_dict['test'] = list(test_df['filepath'])

    for fp in list(test_df['filepath']):  # copy the test images into the folder
        os.makedirs(f'{copy_to}/{fp.split("/")[0]}', exist_ok=True)
        shutil.copyfile(f'../../mpic_data/imgs/{fp}', f'{copy_to}/{fp}')

    tools.write_json(splits_dict, f'../data/{model_names[i]}.json')

for dir in copy_to_list:  # testing if all dirs have the same # of images
    print(dir)
    all_filenames = []
    classes = [c for c in os.listdir(dir) if os.path.isdir(f'{dir}/{c}')]
    for c in classes:
        filenames = [f for f in os.listdir(f'{dir}/{c}') if f != '.DS_Store']
        all_filenames.extend(filenames)
    print(len(all_filenames))