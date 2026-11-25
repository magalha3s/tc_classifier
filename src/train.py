# External Libs
import torch
from dataloaders.shenzendataloader import ShenzenDataLoader
from rich.progress import Progress

# Local Files
from . import utils
from . import tracker

utils.print_ascii()

# Hyperparameters
LR                 = 0.001
BATCH_SIZE         = 32
PATIENCE           = 5

# Model
print("\033[32m Model Setup Started\033[0m")
model     = utils.get_model()
optimizer = torch.optim.Adam(params = model.classifier.parameters(), lr = LR)
criterion  = torch.nn.BCEWithLogitsLoss()

Dataloader = ShenzenDataLoader(batch_size = 64,
                               lazy_load = True,
                               shuffle = True                               
)

patience_limit = 0

for itest in Dataloader:
    for Fold in itest:
        epoch = 0
        X_validation, y_validation = Fold["validation"]
        i_fold, j_fold = Fold.get_id()
        while patience_limit <= PATIENCE:
            for batch, X_train, y_train in enumerate(Fold["train"]):
                model.train()

                # Forward Pass
                y_logits = model.forward(X_train)
                y_pred = torch.sigmoid(y_logits)

                # Calculate Loss
                loss = criterion(y_pred, y_train)

                # Optimizer zero grad
                optimizer.zero_grad()

                # Loss backwards
                loss.backward()

                # Optimizer Step
                optimizer.step()

            avg_loss = 0
            tracker.write_track_file(i_fold = i_fold,
                                     j_fold = j_fold,
                                     data_type = "train",
                                     epoch = epoch,
                                     loss = avg_loss,
                                     sensitivity = 0,
                                     specificity = 0,
                                     sp_value = 0)

            with torch.inference_mode():
                    y_logits_validation = model.forward(X_validation)
                    y_pred_validation = torch.sigmoid(y_logits)

                    # Calculate Loss
                    loss = criterion(y_pred_validation, y_train)
        epoch += 1