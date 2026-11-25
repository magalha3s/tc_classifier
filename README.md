# TC Classifier

[![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)](https://www.python.org/) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)

A Python project for fine-tuning the TorchXRayVision CheXpert DenseNet model for binary chest X-ray classification.

The project loads chest X-ray images, prepares them in the format expected by TorchXRayVision, freezes the pretrained DenseNet backbone, and replaces the classifier head with a single-output binary classifier.

## Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/magalha3s/tc_classifier.git
cd tc_classifier
chmod +x setup.sh
./setup.sh
```

Activate the environment in a later terminal session:

```bash
source .venv/bin/activate
```

## Dataset

The dataset is described by a CSV file and a serialized pickle file containing the cross-validation splits.

### CSV file

The CSV must contain `image_path` and `target` columns:

```csv
,image_path,target
0,case_001.png,True
1,case_002.png,False
```

Labels must use `True` or `False` exactly. `makeFrame` resolves each image relative to the configured image folder and loads the file by its basename.

### Image folder

Place images in a directory and pass that directory to `FoldDataLoader` through `image_folder`.

For example:

```text
project/
├── dataset.csv
├── splits.pkl
└── images/
    ├── case_001.png
    └── case_002.png
```



## Data loading

`FoldDataLoader` can be used to create a custom dataloader per dataset:

```python
from dataloader import FoldDataLoader

class ShenzenDataLoader(FoldDataLoader):
    def __init__(*args):
        super().__init__(args)
        self.splits_path = Path("path/to/splits")
        self.csv_path = Path("path/to/csv")

        # Optional
        self.image_folder = Path("image/folder/name")

```

Training batches are returned as `(images, labels)` tensors. validation and test tensors are also exposed as complete datasets.

## Model

`utils.get_model()` downloads or loads the pretrained TorchXRayVision CheXpert weights and replaces the original classifier with:

```python
torch.nn.Linear(1024, 1)
```

The pretrained backbone is frozen, so only the new classifier head is trainable:

```python
import utils

model = utils.get_model(apply_sigmoud=False)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()
```


## Training

Run the available training entry point with:

```bash
python train.py
```

The current entry point initializes:

- Learning rate: `0.001`
- Patience: `32`
- Epochs: `5`
- Optimizer: Adam
- Loss: binary cross-entropy with logits

Before using the project as a complete training script, configure your dataset paths and fold loop in the application code.



## Project structure

```text
.
├── dataloader.py  # Nested fold and batch data-loading classes
├── io.py          # TorchXRayVision-compatible image transforms
├── setup.sh       # Virtual environment and dependency setup
├── train.py       # Model, optimizer, loss, and training configuration
└── utils.py       # Dataset, metrics, model, and utility helpers
```
