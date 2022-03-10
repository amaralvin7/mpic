import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import torch
from torchvision.models import Inception3
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

# load summary data from model training
with open('modeldata.pkl', 'rb') as file:
    train_acc, val_acc, train_loss, val_loss, test_loader, data = pickle.load(file)

# load an Inception3 model, with structure and weights from saved model
num_classes = len(data.classes)
model = Inception3(init_weights=False)
num_features_aux = model.AuxLogits.fc.in_features
model.AuxLogits.fc = torch.nn.Linear(num_features_aux, num_classes)
num_features_main = model.fc.in_features
model.fc = torch.nn.Linear(num_features_main, num_classes)
model.load_state_dict(torch.load('weights.pt'))
model.eval()

# for plotting sample predictions
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# set seed for reproducible predictions
torch.manual_seed(2)

y_pred = []
y_true = []

with torch.no_grad():

    for i, (inputs, labels) in enumerate(test_loader):
        print(i)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())
        
        if i == 0:
            img_number = 0
            for j in range(inputs.size()[0]):
                fig, ax = plt.subplots()
                ax.axis('off')
                ax.set_title(f'predicted: {data.classes[preds[j]]},\n'
                             f'actual: {data.classes[labels[j]]}')

                inp = inputs.data[j].numpy().transpose((1, 2, 0))
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                plt.imshow(inp)
                fig.savefig(f'./sample_predictions/{img_number}')
                img_number += 1
                plt.close()


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
