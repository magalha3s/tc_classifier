# External Libs
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from rich.progress import Progress
import torchxrayvision as xrv

# Native Libs
from pathlib import Path
import copy
import csv
import math

# Local Files
from . import io_tools

def makeFrame( path : Path, image_folder : Path = Path("images")) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    with open(path, "r") as frame:

        dict_out = {}
        frame = list(csv.reader(frame, delimiter = ','))

        try:
            label_column = frame[0].index('target')
        
        except ValueError as e:
            raise e("'target' row not found in dataset")

        try:
            image_path_column = frame[0].index('image_path')
        
        except ValueError as e:
            raise e("'image_path' row not found in dataset")

        with Progress() as progress:
            load_progress = progress.add_task("[green]Loading Image Folder ...", total = len(frame) - 1)
            try:
                for row in frame[1:]:
                    if row[label_column] not in ["True", "False"]:
                        raise ValueError(f"Invalid value for label '{row[label_column]}' in row '{', '.join(row)}'")

                    with Image.open(path.parent / image_folder / Path(row[image_path_column]).name) as img:
                        img = io_tools.torchxrayvision_transform(img)

                    dict_out[int(row[0])] = (img, torch.tensor(1 if row[label_column] == "True" else 0, dtype = torch.float))
                    

                    progress.update(load_progress, advance = 1)
            
            except ValueError:
                raise ValueError(f"invalid index '{row[0]}' in row '{', '.join(row)}'")
            
            except IndexError as e:
                raise IndexError(f"invalid row '{', '.join(row)}'")
 
            return dict_out

def truth_table(y_preds: torch.Tensor , y_labels: torch.Tensor) -> tuple:
    # TP, FP, TN, FN
    table = {"11":0, "10":0, "00":0, "01":0}
    for pred, label in zip(y_preds, y_labels):
        table[f"{int(pred.item())}{int(label.item())}"]+=1

    return tuple(table.values())


def sensitivity(y_preds: torch.Tensor, y_labels: torch.Tensor) -> float:
    TP, _, _, FN = truth_table(y_preds, y_labels)

    return TP/(TP + FN)


def specificity(y_preds: torch.Tensor, y_labels: torch.Tensor) -> float:
    _, FP, TN, _ = truth_table(y_preds, y_labels)

    return TN/(TN+FP)
    

def sum_product(y_preds: torch.Tensor, y_labels: torch.Tensor) -> float:
    sen, spe = sensitivity(y_preds, y_labels), specificity(y_preds,y_labels)
    
    return math.sqrt(((sen+spe)/2)*math.sqrt(sen*spe))

def set_seeds(seed: int=67) -> None:
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def print_ascii():
    print(r"""

 _____  ___   _____           _       
/__   \/ __\ /__   \_ __ __ _(_)_ __  
  / /\/ /      / /\/ '__/ _` | | '_ \ 
 / / / /___   / /  | | | (_| | | | | |
 \/  \____/   \/   |_|  \__,_|_|_| |_|

""")

def mapToTensor(index : np.int64, info_type: str, frame : dict[int, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    options = {"image" : 0, "label" : 1}

    index = int(index)

    if info_type in options.keys():
        info_type = options[info_type]
    else: 
        raise KeyError(info_type)
    
    return copy.copy(frame[index][info_type])

def mapToTensorListWise(type_list : list[list[np.int64]], frame : dict[int, tuple[torch.Tensor, torch.Tensor]]):
    images = []
    labels = []

    for _type in type_list:
        images_append = []
        labels_append = []
        for index in _type:
            images_append.append(mapToTensor(index = index,
                                            info_type = "image",
                                            frame = frame))

            labels_append.append(mapToTensor(index = index,
                                            info_type = "label",
                                            frame = frame))

        images.append(images_append)
        labels.append(labels_append)

    return images,labels

def model_freeze(model: xrv.models.DenseNet):
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.op_threshs = None

    return model

def get_model(apply_sigmoud: bool = False) -> xrv.models.DenseNet:
    model = model_freeze(xrv.models.DenseNet(weights       = "densenet121-res224-chex",
                                             apply_sigmoid = apply_sigmoud))

    # set seed for control
    set_seeds()
    model.classifier = torch.nn.Linear(1024,1)  # Change classificer head

    return model.to("cuda" if torch.cuda.is_available() else "cpu")