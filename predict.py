import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

import preprocess

# define data and dataloaders
train_path = '/Users/particle/imgs/labeled_grouped_pad/RR'
test_path = '/Users/particle/imgs/labeled_grouped_pad/FK'
# train_path = './labeled_grouped_pad/RR'
# train_path = './labeled_grouped_pad/FK'
train_data = datasets.ImageFolder(train_path)
mean, std = preprocess.get_data_stats(train_path)
transformations = transforms.Compose([  # normalizing with TRAIN stats
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)])
test_data = datasets.ImageFolder(test_path, transformations)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)

# load an resnet18 model, with structure and weights from saved model
num_train_classes = len(train_data.classes)
print(train_data.classes)
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_train_classes)
model.load_state_dict(torch.load('weights.pt'))
model.eval()

# set seed for reproducible predictions
torch.manual_seed(0)

y_pred = []
y_true = []

with torch.no_grad():

    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())


print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
report = classification_report(y_true, y_pred, digits=3, zero_division=0, target_names=data.classes, output_dict=y_true)

# Plot non-normalized confusion matrix
np.set_printoptions(precision=2)
disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=data.classes,
    cmap=plt.cm.Blues,
    normalize='true',
    xticks_rotation='vertical'
)
disp.ax_.set_title('Normalized confusion matrix')
plt.tight_layout()
plt.savefig('confusionmatrix')
plt.close()

#compile data
js_data = pd.read_csv('./data/js_poster_data.csv')
va_data = pd.DataFrame(columns=('label', 'precision', 'recall'))
for label in data.classes:
    precision = report[label]['precision']
    recall = report[label]['recall']
    to_append = [label, precision, recall]
    va_data = va_data.append(pd.Series(to_append, index=va_data.columns), ignore_index=y_true)
        
#merge js and va data
merged = pd.merge(js_data, va_data, on='label')

#make plots
def double_barplot(column):

    labels = list(merged['label'])
    js_vals = merged[f'{column}_x']
    va_vals = merged[f'{column}_y']
    
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots()
    ax.bar(x - width/2, js_vals, width, label='JS', zorder=3)
    ax.bar(x + width/2, va_vals, width, label='VA', zorder=3)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.grid(axis='y', zorder=1)
    ax.set_title(column)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(column)
    plt.close()

double_barplot('precision')
double_barplot('recall')
