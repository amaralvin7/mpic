
import numpy as np
import matplotlib.pyplot as plt
import sys


import torch
from torchvision import datasets, models
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import finetune, preprocess

input_size = 128
batch_size = 128
weights = 'weights_singlestage.pt'

# define data and dataloaders
train_path = '/Users/particle/imgs/relabel_20220926_ttsplit/RR_tvsplit_pad/train'
test_path = '/Users/particle/imgs/relabel_20220926_ttsplit/FK_pad'
train_data = datasets.ImageFolder(train_path)
mean, std = finetune.get_data_stats(train_path, input_size, batch_size)
transformations = finetune.get_resize_transforms(input_size, mean, std)

test_data = datasets.ImageFolder(test_path, transformations)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

# load an resnet18 model, with structure and weights from saved model
num_train_classes = len(test_data.classes)
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_train_classes)
model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
model.eval()

# set seed for reproducible predictions
torch.manual_seed(0)

y_pred = []
y_true = []


with torch.no_grad():

    wrong_counter = 0
    
    for inputs, labels in tqdm(test_loader):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.extend(preds.tolist())
        y_true.extend(labels.tolist())
        
        # for i in range(len(preds)):
        #     if preds[i] != labels[i]:
        #         inp = inputs[i].numpy().transpose((1, 2, 0))
        #         inp = std * inp + mean
        #         inp = np.clip(inp, 0, 1)
        #         plt.imshow(inp)
        #         plt.suptitle(test_data.classes[preds[i]])
        #         preprocess.make_dir(f'wrong_predictions_true_labels/{test_data.classes[labels[i]]}')
        #         plt.savefig(f'wrong_predictions_true_labels/{test_data.classes[labels[i]]}/{wrong_counter}')
        #         plt.close()
        #         wrong_counter += 1


print(classification_report(y_true, y_pred, zero_division=0, target_names=test_data.classes))

# Plot non-normalized confusion matrix
fig, ax = plt.subplots(figsize=(10,10))
disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=test_data.classes,
    cmap=plt.cm.Blues,
    normalize=None,
    xticks_rotation='vertical',
    values_format='.0f',
    ax=ax
)
ax.set_title('Confusion matrix')
plt.tight_layout()
plt.savefig('confusionmatrix')
plt.close()


#make plots
def barplot(metric_vals, metric_name):

    labels = test_data.classes
    x = np.arange(len(labels))  # the label locations
    
    fig, ax = plt.subplots()
    ax.bar(x, metric_vals)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.grid(axis='y', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    fig.savefig(metric_name)
    plt.close()

report = classification_report(y_true, y_pred, zero_division=0, target_names=test_data.classes, output_dict=y_true)
precision = [report[l]['precision'] for l in test_data.classes]
recall = [report[l]['recall'] for l in test_data.classes]
f1 = [report[l]['f1-score'] for l in test_data.classes]
barplot(precision, 'precision')
barplot(recall, 'recall')
barplot(f1, 'f1')
