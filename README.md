# mpic

Marine particle image classification. Images were collected during five different sampling campaigns (i.e., domains), which are referred to as *FC*, *FO*, *JC*, *RR*, and *SR*. Our goal is to show how images collected from a set of domains can be used to predict labels for images from another domain (i.e., out-of-domain or OOD inference). Note that the usage instructions below remain to be elaborated in the future. Training data not yet included.

## Installation (command line)
Conda must be installed ([Miniconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#miniconda-glossary) recommended). After cloning the repository, create a new virtual environment and download all required dependencies:
```
conda env create --name mpic --file environment.yml
```
Install the package locally:
```
pip install -e .
```

## Usage

### Hyperparameter tuning experiments

Tune hyperparameters by using images from *FC*, *FO*, *JC*, and *SR* to train and validate, with RR as the test/target domain.

#### Write train/val/test splits 
```
python ood_splits.py -d RR  # RR is the target domain
python ood_splits.py -d JC  # JC is the target domain (for later use)
```

#### Train model ensembles and use them to predict on the test/target set
```
# 0 is an identifier for models used for hyperparameter tuning
python train_ensembles.py -i 0; python predict_ensembles.py -i 0  
```

### Domain adaptation experiments
Consider a single target domain. Train an out-of-domain model ensemble using images from all other domains. Use this ensemble to predict images from the target domain. Integrate a subset of in-domain predictions into the training set using several different approaches (to be elaborated). Retrain the model ensemble with both in- and out-of-domain images, and predict on any in-domain images not integrated into the testing set.

#### Train OOD model ensemble and use it to predict on the test/target set
```
python train_ensembles.py -i 1
python predict_ensembles.py -i 1  
```
The first line can be omitted if the hyperparameters tuning ensembles were already trained, as this line trains an ensemble already trained during the hyperparameter tuning experiment. "1" is an identifier for considering *RR* as the target domain; use "3" for *JC*.

#### Copy the target domain predictions into a folder
```
python copy_predictions.py -i 1
```
This will create a folder in your data directory called `imgs_from<id>`. "1" is an identifier (`<id>`) for considering *RR* as the target domain; use "3" for *JC*.

#### Manually verify the predictions
Create a copy of `imgs_from<id>` and rename it `imgs_from<id>_verified`. In `imgs_from<id>_verified`, manually verify the predictions by deleting or moving incorrect predictions as appropriate (`imgs_from<id>` will be used as is for testing how much human verification improves inference).

#### Incoporate more instances of minority classes
Create a copy of `imgs_from<id>_verified` and rename it `imgs_from<id>_verified_minboost`. Then,
```
python minboost.py -i 1
```
where "1" is an identifier (`<id>`) for considering *RR* as the target domain; use "3" for *JC*. This will create a folder in your data directory called `imgs_from<id>_minboost` that contains additional suggested images that may belong to the minority classes, i.e., all classes that have less than 100 examples in `imgs_from<id>_verified`. Verify all of the images in `imgs_from<id>_minboost` by deleting those that do not belong in each class folder. Copy the correctly suggested images into `imgs_from<id>_verified_minboost`, then check that no image in that folder has more than one label (again, using `-i 1` for considering *RR* as the target domain and `-i 3` for *JC*):
```
python check_minboost_duplicates.py -i 1
```
#### Write train/val/test splits that include in-domain images
In all code blocks that follow, "2" is an identifier for considering *RR* as the target domain; use "4" for *JC*.
```
python indomain_splits.py -i 2
```

#### Train model ensembles and predict labels for all target domain images not included in the train sets
```
python train_ensembles.py -i 2; python predict_ensembles.py -i 2 
```

### Generate summary figures and text files
```
python figures.py
```

## Acknowledgements
The structure of this repository was inspired by *[The Good Research Code Handbook](https://goodresearch.dev/index.html)* by Patrick Mineault.

## License
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
