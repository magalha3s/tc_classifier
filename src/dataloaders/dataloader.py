# External Libs
import pickle
from pathlib import Path
import torch

# Native Libs
import random
import pickle
from pathlib import Path

# Local Files
from .. import utils

class FoldIter:
    def __init__(self, data : list[torch.Tensor], labels: list[torch.Tensor], batch_size : int, lazy_load : bool = True, shuffle : bool = True) -> None:


        self.data = data
        self.labels = labels
        self.shuffle = shuffle
        self.lazy_load = lazy_load
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if len(self.data) != len(self.labels):
            raise ValueError(f"data and labels size must match")

        if batch_size > len(data):
            raise ValueError(f"Invalid 'batch_size' value") 
        else:
            self.batch_size = batch_size


        # Shuffle tensor list based on a shuffled index list
        if shuffle:

            # ordered index list
            index_list = [i for i in range(len(data))]

            # shuffles the index list
            random.shuffle(index_list)

            # shuffles data and labels
            data = [data[i] for i in index_list]
            labels = [labels[i] for i in index_list]

    
        # Prepare the splits
        data = torch.split(torch.stack(data).to("cpu" if lazy_load == True else self.device), batch_size, 0)
        labels = torch.split(torch.stack(labels).to("cpu" if lazy_load == True else self.device), batch_size, 0)
        
        # Generates the object 
        self.tensor_iterator = list(zip(data, labels))

    # iter method, implements 'for'
    def __iter__(self):
        self.i = -1
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.i+=1
        try:
            if self.lazy_load:
               return self.tensor_iterator[self.i][0].to(self.device), self.tensor_iterator[self.i][1].to(self.device)
            else:
               return self.tensor_iterator[self.i][0], self.tensor_iterator[self.i][1]
        
        except IndexError:
            raise StopIteration

class Fold():
    def __init__(self, data : list[torch.Tensor], labels: list[torch.Tensor], id,lazy_load : bool = True, shuffle : bool = True, batch_size : int = None) -> None:
        self.contents = {"train": FoldIter(data = data[0],
                                           labels = labels[0],
                                           batch_size = batch_size,
                                           lazy_load = lazy_load,
                                           shuffle = shuffle),

                         "validation": (data[1], labels[1]),

                         "test": (data[2], labels[2])
        }
        self.id = id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def __getitem__(self, key : str):
        if key not in self.contents.keys():
            raise KeyError(f"{key}")
        
        if key in ["validation", "test"]:
            return torch.stack(self.contents[key][0]).to(self.device), torch.stack(self.contents[key][1]).to(self.device)
        else: 
            return self.contents[key]

    def get_id(self) -> tuple[int, int]:
        return self.id 

class FoldDataLoader():
    def __init__(self, batch_size : int, lazy_load : bool = False ,shuffle : bool = False) -> None:
        self.splits_path = None
        self.csv_path  = None
        self.image_folder = Path("images")

        if batch_size < 0:
            raise ValueError(f"Invalid 'batch_size' value") 
        else:
            self.batch_size = batch_size

        if type(shuffle) != bool:
            raise TypeError(f"'shuffle' value must be bool")
        else:
            self.shuffle = shuffle

        if type(lazy_load) != bool:
            raise TypeError(f"'lazy_load' value must be bool")
        else:
            self.lazy_load = lazy_load

    def _make_dataset(self) -> None:
        # Check if splits_path, csv_path and image_folder were correctly declared in chield class
        self._check_path_declarations()

        # Loads .pkl file
        if self.splits_path.exists():
            with open(self.splits_path,"rb") as splits_path:
                splits = pickle.load(splits_path)
        else:
            raise FileNotFoundError(f"{self.splits_path}")

        # Object variables
        if self.csv_path.exists():
            frame = utils.makeFrame(path = self.csv_path,
                                 image_folder = self.image_folder)
        else:
            raise FileNotFoundError(f"{self.csv_path}")


        self.dataset = []

        # 0: train, 1: validation, 2: test
        for i, itest in enumerate(splits):
            data_row = []
            for j, jvalidation in enumerate(itest):
                data, labels = utils.mapToTensorListWise(jvalidation,frame)
                data_row.append(Fold(data = data,
                                     labels = labels,
                                     lazy_load = self.lazy_load,
                                     shuffle = self.shuffle,
                                     batch_size = self.batch_size,
                                     id = (i + 1,j + 2)))
                
            self.dataset.append(data_row)

    def __iter__(self) -> list:
        return self.dataset

    def _check_path_declarations(self):
        if self.splits_path == None:
            raise AttributeError("\033[91mError: 'splits_path' must be declared in chield class\033[0m")

        elif self.csv_path == None:
            raise AttributeError("\033[91mError: 'csv_path' must be declared in chield class\033[0m")

        elif self.image_folder == Path("images"):
            print("\033[93mWarn: 'image_folder' not declared, using default value 'images'.\033[0m")
