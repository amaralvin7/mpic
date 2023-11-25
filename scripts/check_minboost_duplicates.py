import os
from collections import Counter

path = '../../mpic_data/imgs_from1_verified_minboost'

classes = [d for d in os.listdir(path) if d != '.DS_Store']

all_fnames = []
for c in classes:
    fnames = [f for f in os.listdir(f'{path}/{c}') if f != '.DS_Store']
    all_fnames.extend(fnames)

duplicates = [k for k,v in Counter(all_fnames).items() if v>1]
print(duplicates)