# TC Classifier

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://www.python.org/) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

A Python project for fine-tuning TorchXRayVision's CheXpert DenseNet model for binary chest X-ray classification.

## Installation

Clone the repository:

```bash
git clone https://github.com/magalha3s/tc_classifier.git
cd tc_classifier
```

Run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

The script creates `.venv`, activates it, installs the project dependencies, and attempts to download model weights.

## Model weights

Before running the project, provide the model weights information in `setup.sh`:

```bash
FILEID="your-google-drive-file-id"
FILENAME="chexpert_weights.pt"
```

Then set the corresponding weight path in `train.py`:

```python
WEIGHTS_PATH = "path_to_zip"
```

> The CheXpert weights must be compatible with the TorchXRayVision DenseNet model used in `model.py`.

## Dataset format

`train.py` expects two inputs:

- `CSV_PATH`: a CSV file indexed with a column named `image_path` and a column named `target`
- `SPLITS_PATH`: a serialized pickle file containing the K-fold splits

Example CSV structure:

```csv
,image_path,target
0,/data/images/case_001.png,0
1,/data/images/case_002.png,1
```

The split structure must produce `K` outer folds and `K - 1` inner partitions for each fold, with train, validation, and test indices available at:

```python
splits[outer_fold][inner_partition][dataset_type]
```

Valid dataset types are:

- `train`
- `validation`
- `test`

## Configuration

Update the placeholders in `train.py`:

```python
LR               = 0.1
BATCH_SIZE       = 32
K                = 10
SPLITS_PATH      = "path/to/splits.pkl"
CSV_PATH         = "path/to/dataset.csv"
WEIGHTS_PATH     = "path/to/weights"
```

The default data loader is initialized with:

```python
Data = K_DataLoader(
    splits_path=SPLITS_PATH,
    csv_path=CSV_PATH,
    K=K,
    batch_size=BATCH_SIZE,
    batch_types=["train", "test", "validation"],
)
```