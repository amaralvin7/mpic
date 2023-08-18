import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report

df = pd.read_csv('./predictions.csv')
labels = yaml.safe_load(open('./config.yaml', 'r'))['classes']

padTrue_cols = [c for c in df.columns if 'padTrue' in c]
padFalse_cols = [c for c in df.columns if 'padFalse' in c]
pred_cols = padTrue_cols + padFalse_cols
metrics = ('precision', 'recall', 'f1-score')
keys = classification_report(df['label'], df[pred_cols[0]], output_dict=True, zero_division=np.nan, labels=labels).keys()
keys = list(keys)
keys.remove('accuracy')
reports = {c: classification_report(df['label'], df[c], output_dict=True, zero_division=np.nan, labels=labels) for c in pred_cols}
print(reports)

fig, axs = plt.subplots(len(metrics), len(keys), tight_layout=True, figsize=(30,10))
for j, k in enumerate(keys):
    axs[-1,j].set_xlabel(k)
    for i, m in enumerate(metrics):
        mean_padTrue = np.nanmean([reports[c][k][m] for c in padTrue_cols])
        std_padTrue = np.nanstd([reports[c][k][m] for c in padTrue_cols], ddof=1)
        mean_padFalse = np.nanmean([reports[c][k][m] for c in padFalse_cols])
        std_padFalse = np.nanstd([reports[c][k][m] for c in padFalse_cols], ddof=1)
        axs[i,j].bar(0, mean_padTrue, yerr=std_padTrue, width=1, color='b', label='padTrue')
        axs[i,j].bar(1, mean_padFalse, yerr=std_padFalse, width=1, color='orange', label='padFalse')
        axs[i,j].set_ylim(-0.1, 1.2)
        axs[i,j].set_xlim(-1, 2)
        axs[i,j].set_xticks([])
        if j == 0:
            axs[i,j].set_ylabel(m)

axs[0,1].legend(frameon=False)

fig.savefig('./figs/metrics.png')

