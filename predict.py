
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, models
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

import finetune

# define data and dataloaders
train_path = '/Users/particle/imgs/labeled_grouped_ttsplit_binary/RR_tvsplit_binary/train'
test_path = '/Users/particle/imgs/labeled_grouped_ttsplit_binary/FK'
train_data = datasets.ImageFolder(train_path)
# mean, std = finetune.get_data_stats(train_path)
mean = [0.303, 0.298, 0.300]
std = [0.104, 0.103, 0.104]
transformations = finetune.get_val_transforms(mean, std)
test_data = datasets.ImageFolder(test_path, transformations)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

# load an resnet18 model, with structure and weights from saved model
num_train_classes = len(train_data.classes)
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_train_classes)
model.load_state_dict(torch.load('weights_binary.pt', map_location=torch.device('cpu')))
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
        
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                inp = inputs[i].numpy().transpose((1, 2, 0))
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                plt.imshow(inp)
                plt.savefig(f'wrong_predictions_true_labels/{test_data.classes[labels[i]]}/{wrong_counter}')
                plt.close()
                wrong_counter += 1


print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
report = classification_report(y_true, y_pred, digits=3, zero_division=0, target_names=test_data.classes, output_dict=y_true)

# Plot non-normalized confusion matrix
np.set_printoptions(precision=2)
disp = ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    display_labels=test_data.classes,
    cmap=plt.cm.Blues,
    normalize='true',
    xticks_rotation='vertical'
)
disp.ax_.set_title('Normalized confusion matrix')
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
    ax.set_title(metric_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60)
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(metric_name)
    plt.close()

precision = [report[l]['precision'] for l in test_data.classes]
recall = [report[l]['recall'] for l in test_data.classes]
barplot(precision, 'precision')
barplot(recall, 'recall')
