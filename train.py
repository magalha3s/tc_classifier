# External libs
import torch

# Local imports
from model import get_model
from helper_functions import sensitivity,specificity,sum_product
from dataloader import K_DataLoader
# HYPERPARAMETERS
LR               = 0.1
BATCH_SIZE       = 32
K                = 10
SPLITS_PATH      = ""
CSV_PATH         = ""
WEIGHTS_PATH     = ""

# Model
model     = get_model(WEIGHTS_PATH)
optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)
loss_fn   = torch.nn.BCELoss()

#Data
Data = K_DataLoader(splits_path  = SPLITS_PATH,
                    csv_path     = CSV_PATH,
                    K            = K,
                    batch_size   = BATCH_SIZE,
                    batch_types  = ["train","test","validation"])