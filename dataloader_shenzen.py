import pickle 
from pathlib import Path
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from rich.progress import Progress

IMAGE_FOLDER = Path("/mnt/shared/storage03/projects/brics/public/Shenzhen/china/raw/images")
SPLITS_PATH = Path("/mnt/shared/storage03/projects/brics/public/Shenzhen/china/raw") / "splits.pkl"
CSV_PATH = Path("/mnt/shared/storage03/projects/brics/public/Shenzhen/china/raw") / "Shenzhen_china_table_from_raw.csv"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

class ShenzenDataLoader():
    def __init__(self, splits_path : Path, csv_path: Path, K: int, batch_size : int, batch_type : str = "train") -> None:

        if not csv_path.exists() or not splits_path.exists():
            raise FileNotFoundError(f" {str(csv_path) if not csv_path.exists() else str(splits_path)} does not exist")
        
        # Check if batch_type is valid
        options = {"train":0, "validation": 1, "test": 2}
        if batch_type not in options.keys():
            raise KeyError(f" '{batch_type}' is an invalid option ")
        else:
            self.batch_type = options[batch_type]

        # Loads .pkl file
        with open(splits_path,"rb") as splits_file:
            self.splits = pickle.load(splits_file)

        # Check if K value is valid
        self.K = K
        if K != len(self.splits):
            raise ValueError(f" '{self.K}' is not equal to dataset's K")
        else: pass

        # Object variables
        self.shenzenFrame   = pd.read_csv(csv_path,index_col=0)[["image_path","target"]]
        self.device         = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_dataset  = []
        self.label_dataset  = []

    def prepare_kfold(self):
        path_column = self.shenzenFrame["image_path"]
        index_image = {}
        with Progress() as progress:

            task1 = progress.add_task("[green] Pre Processing Images...", total=path_column.size)
            task2 = progress.add_task("[red] Folds Processed...", total=self.K*(self.K-1))
        # Create a dict with the images converted to tensors for efficiency
            for index, path in zip(path_column.index, path_column):
                with Image.open(IMAGE_FOLDER / Path(path).name) as img:
                    index_image[index] = transform(img).to(self.device)
                    progress.update(task1,1)

            # Make a shallowcopy of images to reduce storage usage
            for i in range(self.K):
                for j in range(self.K-1):
                    label_n  = []
                    image_n  = []
                    
                    for index in self.splits[i][j][self.batch_type]:
                        image_n.append(copy.copy(index_image[int(index)])) 
                        label_n.append(torch.tensor(self.shenzenFrame["target"].iloc[index],dtype=torch.float))

                    self.label_dataset.append(torch.stack(label_n))
                    self.image_dataset.append(torch.stack(image_n))

                    progress.update(task2,1)

    def __getitem__(self, key: int) -> tuple:
        # Detects invalid indexes
        if key >= len(self.image_dataset):
            raise IndexError(f" '{key}' is not a valid fold ")
        else:
            return self.image_dataset[key], self.label_dataset[key]