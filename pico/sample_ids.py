import os
import csv

f = open('sample_ids.csv','w')
w = csv.writer(f)
for path, _, files in os.walk('/Users/particle/imgs/orig'):
    label = os.path.basename(path)
    for filename in files:
        if '.jpg' in filename:
            w.writerow([filename, label])