import os
import pandas as pd
import matplotlib.pyplot as plt

rr_path = '/Users/particle/imgs/relabel_20220926_ttsplit/RR/'
fk_path = '/Users/particle/imgs/relabel_20220926_ttsplit/FK/'
contents = os.listdir(rr_path)
labels = [name for name in contents if os.path.isdir(os.path.join(rr_path, name))]
labels.sort()

def count_samples(path, labels):
    
    data_set = os.path.basename(path[:-1])
    total = 0
    counts = []
    for l in labels:
        n_smps = len([f for f in os.listdir(f'{path}{l}') if f.endswith('.jpg')])
        counts.append(n_smps)
        total += n_smps
    count_df = pd.DataFrame(counts, index=labels, columns=[data_set])
    return count_df, total

rr_df, rr_total = count_samples(rr_path, labels)
fk_df, fk_total = count_samples(fk_path, labels)
df = pd.concat([rr_df, fk_df], axis=1)

fig = plt.figure() # Create matplotlib figure

ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.

width = 0.2

# df['RR'].plot(kind='bar', color='b', ax=ax, width=width, position=1)
# df['FK'].plot(kind='bar', color='orange', ax=ax2, width=width, position=0)

# ax.set_ylabel('RR')
# ax2.set_ylabel('FK')

df.plot(kind= 'bar', secondary_y='RR')

plt.suptitle(f'RR total: {rr_total}, FK total: {fk_total}')
plt.tight_layout()
plt.savefig('n_samples')
plt.close()
