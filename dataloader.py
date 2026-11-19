# Internal libs
import pickle
import copy 
from pathlib import Path
from typing import Annotated

# External libs
import pandas as pd
from PIL import Image
import torch
import copy
import torchxrayvision as xrv

# local modules
import transform_compositions as tc

# Get item with K_DataLoader[key,fold], 
# "key" can be "test" or "train" or "validation" and needs to be loaded in "batch_types" 
# "fold" varies in the range [0, K*(K-1)[

class K_DataLoader():
    def __init__(self, splits_path : Path, csv_path: Path,K: int, batch_size : int, batch_types : list[str], image_folder: Path = None) -> None:

        # Check files existence
        if not csv_path.exists() or not splits_path.exists():
            raise FileNotFoundError(f" {str(csv_path) if not csv_path.exists() else str(splits_path)} does not exist")

        # Check for custom image path
        if image_folder != None:
            if not image_folder.exists():
                raise FileNotFoundError(f" {str(image_folder)} does not exist")
            else:
                self.image_folder = image_folder
            
        # Check if batch_type is valid
        self.options    = {"train":0, "validation": 1, "test": 2}
        self.batch_types = []
        for _type in batch_types:
            if _type not in self.options.keys():
                raise KeyError(f" '{_type}' is an invalid option ")
            else:
                self.batch_types.append(_type)

        if batch_size <= 0:
            raise ValueError(f" '{batch_size}' is not a valid batch size")
        else:
            self.batch_size = batch_size

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
        self.image_dataset  = {"train":[], "test":[], "validation":[]}
        self.label_dataset  = {"train":[], "test":[], "validation":[]}

        self._prepare_kfold()

    def _prepare_kfold(self) -> None:
        path_column = self.shenzenFrame["image_path"]
        index_image = {}

        # Create a dict with the images converted to tensors for efficiency, avoiding to reload the image dict every fold iteration
        for index, path in zip(path_column.index, path_column):
            try:
                with Image.open(self.image_folder / Path(path).name if self.image_folder != None else Path(path)) as img:
                    index_image[index] = tc.default_transform(img).to("cpu")
            except FileNotFoundError:
                raise FileNotFoundError(f"no image found in {path}")

        # Make a shallowcopy of images to use them as pointers
        # May need to adujust .to(device) settings to "cpu" and send to "cuda" by batch to reduce VRAM usage (this will decrease training speed).
        for _type in self.batch_types: 
            for i in range(self.K):
                for j in range(self.K-1):
                    label_n  = []
                    image_n  = []
                    
                    for index in self.splits[i][j][self.options[_type]]:
                        image_n.append(copy.copy(index_image[int(index)])) 
                        label_n.append(torch.tensor(self.shenzenFrame["target"].loc[index],dtype=torch.float))

                    self.label_dataset[_type].append(label_n)
                    self.image_dataset[_type].append(image_n)

    def __getitem__(self, key : tuple[str, int]) -> tuple[torch.Tensor, torch.Tensor] | tuple[tuple[torch.Tensor],tuple[torch.Tensor]]:
        
        # Detects invalid indexes
        if key[0] not in self.batch_types:
            raise KeyError(f" '{key[0]}' is not a valid key or wasn't loaded")

        # Only training data is sent in split format
        else:
            X, y = torch.stack(self.image_dataset[key[0]][key[1]]).to(self.device), torch.stack(self.label_dataset[key[0]][key[1]]).to(self.device)

            if key[0] == "train":

                # Return a tuple of tuples of batches and batch labels with size "batch_size", ( (batch1,batch1_labels), (batch2,batch2_labels) ... )
                return zip(torch.split(X,self.batch_size), torch.split(y,self.batch_size))
            else:
                # Return a tuple of images and labels,  (images,label)
                return X,y

    def __len__(self):
        return len(self.shenzenFrame.index)