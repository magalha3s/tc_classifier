import torchxrayvision as xrv
import torch
from helper_functions import set_seeds

def model_freeze(model: torch.nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model

def get_model() -> torch.nn.Module:
    model = model_freeze(xrv.models.DenseNet(weights="densenet121-res224-chex"))

    # set seed for control
    set_seeds(67)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(1024,1),
                                           torch.nn.Sigmoid())   # Change classificer head

    return model