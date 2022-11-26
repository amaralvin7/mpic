import re
import os


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import numpy as np
import pandas as pd
import yaml

from colors import *

path = '/Users/particle/imgs/relabel_20221110'
contents = os.listdir(path)
labels = [name for name in contents if os.path.isdir(os.path.join(path, name))]
labels.sort()

def count_samples(domain):
    
    get_domain = re.compile('.+?(?=\d)')
    total = 0
    counts = []
    for l in labels:
        n_smps = 0
        for f in os.listdir(os.path.join(path, l)):
            ext_ok = f.split('.')[1] in ('jpg', 'tiff')
            if ext_ok and get_domain.search(f).group() == domain:
                n_smps += 1
        counts.append(n_smps)
        total += n_smps
    count_df = pd.DataFrame(counts, index=labels, columns=[domain])

    return count_df, total

cfg = yaml.safe_load(open('config.yaml', 'r'))

rr_df, rr_total = count_samples('RR')
fk_df, fk_total = count_samples('FK')
srt_df, srt_total = count_samples('SRT')
df = pd.concat([rr_df, fk_df, srt_df], axis=1)
# df = df.sort_values('RR', ascending=False)

print(df)
for c in df.index:
    if c not in cfg['classes']:
        df.drop(labels=c, inplace=True)
        
#normalize
new_rr_total = df['RR'].sum()
new_srt_total = df['SRT'].sum()
new_fk_total = df['FK'].sum()
df['RR'] = df['RR']/new_rr_total
df['SRT'] = df['SRT']/new_srt_total
df['FK'] = df['FK']/new_srt_total

# barplot
ind = np.arange(len(df)) 
width = 0.25
bar1 = plt.bar(ind, df['RR'], width, color=blue)
bar2 = plt.bar(ind+width, df['SRT'], width, color=green)
bar3 = plt.bar(ind+width*2, df['FK'], width, color=orange)

# plt.yscale('log')
plt.ylabel('Fraction of observations')
plt.subplots_adjust(bottom=0.2)
plt.xticks(ind+width, df.index.values)
plt.xticks(rotation=45, ha='right')
plt.legend((bar1, bar2, bar3), ('RR', 'SRT', 'FK'), ncol=3, bbox_to_anchor=(0.5, 1.02), loc='lower center',
            handletextpad=0.1, frameon=False)
plt.show()
