'''
Compare agreement between Colleen's two labeling attempts of a subset of
30k images from the RR set.
'''
import pandas as pd

df = pd.read_csv('../data/relabels.csv', index_col=0)
n_matches = len(df[df.nunique(axis=1) == 1])
relabel_rate = n_matches / len(df)
print(relabel_rate)
