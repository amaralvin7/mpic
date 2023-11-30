import argparse
import os
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_list_id', '-i')
args = parser.parse_args()
list_id = int(args.model_name_list_id)

path = f'../../mpic_data/imgs_from{list_id}_verified_minboost'

classes = [d for d in os.listdir(path) if d != '.DS_Store']

all_fnames = []
for c in classes:
    fnames = [f for f in os.listdir(f'{path}/{c}') if f != '.DS_Store']
    all_fnames.extend(fnames)

duplicates = [k for k,v in Counter(all_fnames).items() if v>1]
print(duplicates)