from pathlib import Path
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


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

def plot_roc_curve(y_preds : list, y_labels : list) -> None:
    fig, ax = plt.subplots()
    t_rates = np.arange(0, 1, 0.05)

    x = []
    y = []
    for rate in t_rates:
        for index, pred in enumerate((n_y_preds := copy.deepcopy(y_preds))):
            n_y_preds[index] = 1 if pred>=rate else 0

        sp = specificity(n_y_preds, y_labels)
        sv = sensitivity(n_y_preds, y_labels)


        x.append(1-sp)
        y.append(sv)

    # Title

    ax.plot(x,y)
    ax.set_title("ROC Curve")
    ax.set_ylabel("TPR (Sensitivity)")
    ax.set_xlabel("FPR (Specificity)")

    # Style
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    plt.show()

def set_seeds(seed: int=67):
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def print_ascii():
    path = Path().cwd() / "ascii.txt"
    if path.exists():
        with open(path,"r") as file:
            for line in file.readlines():
                print(line,end="")
    print("\n--------------------------------------")