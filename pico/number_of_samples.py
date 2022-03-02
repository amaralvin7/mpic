import os

path = './data/by_label/'
contents = os.listdir(path)
labels = [name for name in contents if os.path.isdir(f'{path}{name}')]
labels.sort()

total = 0
for l in labels:
    n_smps = len([f for f in os.listdir(f'{path}{l}') if f.endswith('.jpg')])
    total += n_smps
    print(f'{l}: {n_smps}')
print(f'TOTAL: {total}')