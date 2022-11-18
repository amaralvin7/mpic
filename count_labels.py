import re
import os


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import numpy as np
import pandas as pd

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

rr_df, rr_total = count_samples('RR')
fk_df, fk_total = count_samples('FK')
srt_df, srt_total = count_samples('SRT')
df = pd.concat([rr_df, fk_df, srt_df], axis=1)
df = df.sort_values('RR', ascending=False)

print(df)
df.drop(labels='skip', inplace=True)

# barplot
ind = np.arange(len(df)) 
width = 0.25
bar1 = plt.bar(ind, df['RR'], width, color='b')
bar2 = plt.bar(ind+width, df['FK'], width, color='g')
bar3 = plt.bar(ind+width*2, df['SRT'], width, color='orange')

plt.yscale('log')
plt.subplots_adjust(bottom=0.2)
plt.xticks(ind+width, df.index.values)
plt.xticks(rotation=45, ha='right')
plt.legend( (bar1, bar2, bar3), ('RR', 'FK', 'SRT') )
plt.show()
