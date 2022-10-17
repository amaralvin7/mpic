
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, models
from tqdm import tqdm
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import finetune, preprocess

input_size = 128
batch_size = 128
weights = 'weights.pt'

# define data and dataloaders
train_path = '/Users/particle/imgs/relabel_20221014_sitesplit/RR_tvsplit_pad/train'
test_path = '/Users/particle/imgs/relabel_20221014_sitesplit/FK_pad'
train_data = datasets.ImageFolder(train_path)
mean, std = finetune.get_data_stats(train_path, input_size, batch_size)
transformations = finetune.to_tensor(mean, std)

num_train_classes = len(train_data.classes)
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_train_classes)
model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
model.eval()


class UnlabeledImageFolder(torch.utils.data.Dataset):

    def __init__(self, dirpath, transform):
        super().__init__()

        filenames = os.listdir(dirpath)
        self.imagepaths = [os.path.join(dirpath, f) for f in filenames if '.jpg' in f]
        self.transform = transform

    def __getitem__(self, index):
        imagepath = self.imagepaths[index]
        filename = os.path.basename(imagepath)

        with Image.open(imagepath) as image:
            image = image.convert('RGB')

        image = self.transform(image)

        return filename, image

    def __len__(self):
        return len(self.imagepaths)


def get_test_loader(data_object, test_path):
    
    test_data = data_object(test_path, transformations)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return test_loader


def predict_with_truth():
    
    test_loader = get_test_loader(datasets.ImageFolder, test_path)
    
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
                    plt.suptitle(train_data.classes[preds[i]])
                    preprocess.make_dir(f'wrong_predictions_true_labels/{train_data.classes[labels[i]]}')
                    plt.savefig(f'wrong_predictions_true_labels/{train_data.classes[labels[i]]}/{wrong_counter}')
                    plt.close()
                    wrong_counter += 1

    print(classification_report(y_true, y_pred, zero_division=0, target_names=train_data.classes))

    _, ax = plt.subplots(figsize=(10,10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=train_data.classes,
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

    return y_pred, y_true


def predict_without_truth():
    
    test_loader = get_test_loader(UnlabeledImageFolder, test_path)
    
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        
        for filenames, inputs in tqdm(test_loader):
            outputs = model(inputs)
            _, labels = torch.max(outputs, 1)
            all_labels.extend([train_data.classes[i] for i in labels.tolist()])
            all_filenames.extend(filenames)
    
    df = pd.DataFrame(list(zip(all_filenames, all_labels)), columns =['file', 'predicted_label'])
    df.to_csv('predictions.csv', index=False)


def barplots(y_true, y_pred):  # for precision, recall, f1 score

    def make_plot(metric_vals, metric_name):
        labels = train_data.classes
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

    report = classification_report(y_true, y_pred, zero_division=0, target_names=train_data.classes, output_dict=y_true)
    precision = [report[l]['precision'] for l in train_data.classes]
    recall = [report[l]['recall'] for l in train_data.classes]
    f1 = [report[l]['f1-score'] for l in train_data.classes]
    make_plot(precision, 'precision')
    make_plot(recall, 'recall')
    make_plot(f1, 'f1')

if __name__ == '__main__':
    y_pred, y_true = predict_with_truth()
    barplots(y_true, y_pred)