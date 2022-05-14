import os
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/particle/imgs/orig/'
contents = os.listdir(path)
labels_all = [name for name in contents if os.path.isdir(f'{path}{name}')]


def count_labels(labels, site='_'):
    
    # print(f'-------------{site}-------------')
    
    n_per_label = []
    labels.sort()
    total = 0
    
    for l in labels:
        n_smps = len([f for f in os.listdir(f'{path}{l}') if f.endswith('.jpg') and site in f])
        n_per_label.append(n_smps)
        total += n_smps
    #     print(f'{l}: {n_smps}')
    # print(f'TOTAL: {total}')
    
    return n_per_label

all_counts = count_labels(labels_all)
fk_counts = count_labels(labels_all, 'FK')
rr_counts = count_labels(labels_all, 'RR')

# n_per_label_normed = [n/total for n in n_per_label]
# y_pos = range(len(labels_all))
# plt.bar(y_pos, n_per_label)
# plt.xticks(y_pos, labels_all, rotation=90)
# plt.tight_layout()
# plt.savefig('sample_counts')

fig, ax = plt.subplots(tight_layout=True)
x_axis = np.arange(len(labels_all))
ax.bar(x_axis -0.2, fk_counts, width=0.4, label = 'FK')
ax.bar(x_axis +0.2, rr_counts, width=0.4, label = 'RR')
ax.set_xticks(x_axis, labels_all, rotation=90)
ax.legend()
plt.show()