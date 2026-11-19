# External libs
import torch

# Local imports
from chex_model import get_model
from helper_functions import sensitivity,specificity,sum_product
from dataloader import K_DataLoader
# HYPERPARAMETERS
LR               = 0.001
BATCH_SIZE       = 32
EPOCHS           = 5
K                = 10
SPLITS_PATH      = ""
CSV_PATH         = ""

# Model
model     = get_model()
optimizer = torch.optim.Adam(params = model.parameters(), lr = LR)
loss_fn   = torch.nn.BCELoss()

#Data
Data = K_DataLoader(splits_path  = SPLITS_PATH,
                    csv_path     = CSV_PATH,
                    K            = K,
                    batch_size   = BATCH_SIZE,
                    batch_types  = ["train","test","validation"])




for epoch in EPOCHS:
    for X_train, y_train in Data["train", 0]:
        model.train()
        
        y_pred = model.forward(X_train)

        loss = loss_fn(y_pred,y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.inference_mode(): 
        for X_validation, y_validation in Data["validation", 0]:
            
            # 1. Forward pass
            validation_pred = model(X_validation)
            
            # 2. Calculate loss and accuracy