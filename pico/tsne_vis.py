import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

filename = 'features.h5'
kelly_colors = ['#F2F3F4', '#222222', '#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280', '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522', '#E25822', '#2B3D26']

with h5py.File(filename, "r") as f:
    
    ids = list(f['object_id'])
    features = list(f['features'])

labels_all = [str(i).split('_')[-1][:-1] for i in ids]
unique_labels = sorted(list(set(labels_all)))

colors = kelly_colors[1:len(unique_labels)+1]
legend_entries = dict(zip(unique_labels,colors))
label_colors = [legend_entries[l] for l in labels_all]

x = [i[0] for i in features]
y = [i[1] for i in features]

fig, ax = plt.subplots()
ax.scatter(x, y, label=labels_all, c=label_colors)

leg_elements = []
for label, color in legend_entries.items():
    leg_elements.append(
        Line2D([0], [0], marker='o', c='white', label=label,
        markerfacecolor=color, ms=9, frameon=False)
    )

ax.legend(handles=leg_elements, ncol=1)

plt.show()
    