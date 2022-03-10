import os
import matplotlib.pyplot as plt

path = './data/by_label/'
contents = os.listdir(path)
labels = [name for name in contents if os.path.isdir(f'{path}{name}')]
n_per_label = []
labels.sort()

total = 0
for l in labels:
    n_smps = len([f for f in os.listdir(f'{path}{l}') if f.endswith('.jpg')])
    n_per_label.append(n_smps)
    total += n_smps
    print(f'{l}: {n_smps}')
print(f'TOTAL: {total}')

# n_per_label_normed = [n/total for n in n_per_label]

y_pos = range(len(labels))
plt.bar(y_pos, n_per_label)
plt.xticks(y_pos, labels, rotation=90)
plt.tight_layout()
plt.savefig('sample_counts')