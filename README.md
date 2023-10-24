# mpic

Marine particle image classification. Training data not yet included.

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

1. Train model ensembles:
```
cd scripts
python train_ensembles -c hptune
```

2. Evaluate the trained models:
```
python hptune_predict.py
```

3. Generate figures and a text file with evaluation metrics:
```
python figures.py
```

## Acknowledgements
The structure of this repository was inspired by *[The Good Research Code Handbook](https://goodresearch.dev/index.html)* by Patrick Mineault.

## License
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
