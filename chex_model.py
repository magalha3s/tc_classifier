import torchxrayvision as xrv
import torch
from helper_functions import set_seeds

def model_freeze(model: xrv.models.DenseNet):
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.op_threshs = None

    return model

def get_model(apply_sigmoud: bool = False) -> xrv.models.DenseNet:
    model = model_freeze(xrv.models.DenseNet(weights       = "densenet121-res224-chex",
                                             apply_sigmoid = apply_sigmoud))

    # set seed for control
    model.classifier = torch.nn.Linear(1024,1)  # Change classificer head

    return model