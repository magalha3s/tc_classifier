import torchxrayvision as xrv
import torch
from helper_functions import set_seeds

def model_freeze(model: torch.nn.Module):
    for parameter in model.feature.parameters():
        parameter.requires_grad = False

    return model

def get_model(weights_path: str) -> torch.nn.Module:
    model = model_freeze(xrv.baseline_models.chexpert.DenseNet(weights_path))

    # set seed for control
    set_seeds(67)

    model.classifier = torch.nn.Sequential(torch.nn.Linear(1024,1),
                                           torch.nn.Sigmoid(1,1))   # Change classificer head

    return model