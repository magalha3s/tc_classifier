# TC Classifier

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![License](https://img.shields.io/badge/license-not%20specified-lightgrey)](#license)

A Python project for fine-tuning TorchXRayVision's CheXpert DenseNet model for binary chest X-ray classification.

The repository currently provides the model wrapper, image preprocessing, K-fold data loading, and training configuration foundations. It is under active development.

## Features

- Loads a TorchXRayVision CheXpert DenseNet model
- Freezes the pretrained feature extractor
- Replaces the classifier head with a sigmoid binary classifier
- Preprocesses images by resizing to `256 × 256`, converting to grayscale, and creating a tensor
- Loads train, validation, and test folds from a pickled split file
- Supports CUDA when available, with automatic fallback to CPU
- Includes sensitivity, specificity, combined-score, and ROC plotting utilities
- Provides a virtual environment and dependency setup script

## Project structure

| File | Description |
| --- | --- |
| `model.py` | Creates the CheXpert model and replaces its classifier head |
| `dataloader.py` | Defines `K_DataLoader` for CSV metadata and K-fold image loading |
| `transform_compositions.py` | Defines the default TorchVision image preprocessing pipeline |
| `helper_functions.py` | Contains metrics, seed control, and ROC plotting helpers |
| `train.py` | Defines hyperparameters and initializes the model and data loader |
| `setup.sh` | Creates a virtual environment, installs dependencies, and downloads model weights |

## Requirements

- Python 3.10 or newer
- Linux/macOS shell for `setup.sh`
- PyTorch with an optional compatible CUDA installation
- A dataset CSV containing image paths and binary targets
- A pickle file containing the K-fold split structure

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

To set up manually:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision matplotlib numpy pandas Pillow torchxrayvision gdown
```

## Model weights

Before running the project, provide the model weights information in `setup.sh`:

```bash
FILEID="your-google-drive-file-id"
FILENAME="chexpert_weights.pt"
```

Then set the corresponding weight path in `train.py`:

```python
WEIGHTS_PATH = "chexpert_weights.pt"
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

## Usage

After filling in the dataset and weights paths, run:

```bash
python train.py
```

Accessing a loaded partition through the data loader uses:

```python
train_images, train_labels = Data["train", fold_index]
validation_images, validation_labels = Data["validation", fold_index]
test_images, test_labels = Data["test", fold_index]
```

Training batches from a train partition are split according to `BATCH_SIZE`:

```python
train_batches, label_batches = Data["train", fold_index]
```

## Metrics

The helper module provides:

- `sensitivity(y_preds, y_labels)`
- `specificity(y_preds, y_labels)`
- `sum_product(y_preds, y_labels)`
- `plot_roc_curve(specificity, sensitivity)`
- `set_seeds(seed)`

## Image preprocessing

Every image is transformed with:

```python
transforms.Resize((256, 256))
transforms.Grayscale(num_output_channels=1)
transforms.ToTensor()
```

The data loader loads each image once, converts it to a tensor, and reuses references across the generated folds.

## Current limitations

This is an early-stage implementation. The following areas are not yet complete:

- The setup script contains empty `FILEID` and `FILENAME` placeholders
- The training loop is not implemented in `train.py`
- Dataset and split path variables are empty placeholders
- The full K-fold training/evaluation workflow is incomplete
- Prediction thresholding and binary output handling need to be finalized
- A formal test suite is not included
- No license is currently specified

## Contributing

Contributions are welcome. When modifying the project, keep changes focused and include a clear description of the model, data, or metrics affected.

## License

No license has been specified for this repository. All rights remain with the repository owner until a license is added.
