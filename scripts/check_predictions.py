import pandas as pd

for i in range(5):
    old_df = pd.read_csv(f'../results/hitloopI/predictions/A-{i}.csv')
    new_df = pd.read_csv(f'../results/predictions/targetRR_ood-{i}.csv')
    print(old_df.equals(new_df))