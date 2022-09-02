
import matplotlib.pyplot as plt
import pandas as pd
import sys

from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv('Compared_classification.csv')

# print(df.head())
# sys.exit()

df['ID_xg'] = df['ID_x']
df['ID_yg'] = df['ID_y']

sphere_aggregate = ('aggregate', 'dense_detritus', 'mini_pellet')
cylinder_pellet = ('large_loose_pellet', 'long_fecal_pellet')
swimmer = ('zooplankton', 'zooplankton_part', 'copepod', 'amphipod', 'pteropod')

for col in ('ID_x', 'ID_y'):
    df.loc[df[col].isin(sphere_aggregate), f'{col}g'] = 'sphere_aggregate'
    df.loc[df[col].isin(cylinder_pellet), f'{col}g'] = 'cylinder_pellet'
    df.loc[df[col].isin(swimmer), f'{col}g'] = 'swimmer'

y1 = df['ID_xg']
y2 = df['ID_yg']

recovery_ungrouped = sum(df['ID_x'] == df['ID_y'])/len(df)*100
recovery_grouped = sum(df['ID_xg'] == df['ID_yg'])/len(df)*100

print(f'Recovery ungrouped: {recovery_ungrouped:.0f}')
print(f'Recovery grouped: {recovery_grouped:.0f}')

# fig, ax = plt.subplots(figsize=(10,10))
# disp = ConfusionMatrixDisplay.from_predictions(
#     y_true=y1,
#     y_pred=y2,
#     # display_labels=test_data.classes,
#     cmap=plt.cm.Blues,
#     normalize='true',
#     xticks_rotation='vertical',
#     values_format='.2f',
#     ax=ax
# )
# # ax.set_title('Normalized confusion matrix')
# plt.tight_layout()
# plt.savefig('label_comparison_normalized')
# plt.close()
