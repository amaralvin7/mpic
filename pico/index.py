import os
import csv

# w.writerow(['object_id', 'label', 'path'])
# for path, _, files in os.walk('/Users/particle/imgs/orig'):
#     label = os.path.basename(path)
#     for filename in files:
#         if '.jpg' in filename:
#             filepath = f'{label}/{filename}'
#             w.writerow([filename, label, filepath])

data_dir = '/Users/particle/imgs'
filenames = [f for f in os.listdir(f'{data_dir}/padded') if f.endswith('.jpg')]

f = open(f'{data_dir}/index.csv','w')
w = csv.writer(f)
w.writerow(['object_id', 'path'])
for filename in filenames:
    w.writerow([filename.split('.')[0], f'padded/{filename}'])