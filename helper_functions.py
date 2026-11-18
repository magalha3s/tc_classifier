import torch
import math
import matplotlib.pyplot as plt
import numpy as np

def truth_table(y_preds: torch.Tensor , y_labels: torch.Tensor) -> tuple:
    # TP, FP, TN, FN
    table = {"11":0, "10":0, "00":0, "01":0}
    for pred, label in zip(y_preds, y_labels):
        table[f"{pred.item()}{label.item()}"]+=1

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

def plot_roc_curve(specificity : list, sensitivity : list) -> None:
    fig, ax = plt.subplots()

    # Plot
    ax.plot(specificity, sensitivity, label="Mean ROC")

    # Title
    ax.set_title("ROC Curve")
    ax.set_ylabel("TPR (Sensitivity)")
    ax.set_xlabel("FPR (Specificity)")

    # Style
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))

    ax.grid(True)
    ax.legend()

    plt.show()

def model_refactor_tc(model: torch.nn.Module):
    for parameter in model.feature.parameters():
        parameter.requires_grad = False
    