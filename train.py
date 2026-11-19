# External libs
import torch
from pathlib import Path
from tqdm import tqdm

# Local imports
from chex_model import get_model
from helper_functions import print_ascii,sensitivity,specificity,sum_product
from dataloader import K_DataLoader

# Print ascii info
print_ascii()

# HYPERPARAMETERS
LR               = 0.001
BATCH_SIZE       = 32
EPOCHS           = 5
K                = 10
SPLITS_PATH      = Path("/mnt/shared/storage03/projects/brics/public/Shenzhen/china/raw/splits.pkl")
CSV_PATH         = Path("/mnt/shared/storage03/projects/brics/public/Shenzhen/china/raw/Shenzhen_china_table_from_raw.csv")
IMAGE_FOLDER     = Path("/mnt/shared/storage03/projects/brics/public/Shenzhen/china/raw/images") # Change if root isn't the same as the ones in the csv file, None if not



# Model
print("\033[32m Model Setup Started\033[0m")
model     = get_model()
optimizer = torch.optim.Adam(params = model.classifier.parameters(), lr = LR)
criterion  = torch.nn.BCEWithLogitsLoss()

#Data
print("\033[32m Data Setup Started\033[0m")
Data = K_DataLoader(splits_path   = SPLITS_PATH,
                    csv_path      = CSV_PATH,
                    K             = K,
                    batch_size    = BATCH_SIZE,
                    batch_types   = ["train","test","validation"],
                    image_folder = IMAGE_FOLDER)



# Training Informations
avg_loss_per_epoch = []
avg_val_loss_per_epoch = []

for epoch in range(EPOCHS):
    train_loss = 0

    for X_train, y_train in tqdm(Data["train", 0], desc=f"Epoch {epoch+1}/{EPOCHS}"):
        model.train()

        y_pred = model.forward(X_train)

        loss = criterion(y_pred.squeeze(1), y_train)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss_per_epoch.append(train_loss / len(list(Data["train", 0])))

    # Validation
    model.eval()
    val_loss = 0


    with torch.no_grad():
        X_val, y_val = Data["validation", 0]

        y_pred = model(X_val)

        loss = criterion(y_pred.squeeze(1), y_val)

        val_loss = loss.item()

    avg_val_loss_per_epoch.append(val_loss)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_loss_per_epoch[-1]:.4f} | "
        f"Val Loss: {avg_val_loss_per_epoch[-1]:.4f}"
    )

# Test fold 0 evaluation
model.eval()

test_loss = 0

with torch.no_grad():

    X_test, y_test = Data["test", 0]

    y_pred = model(X_test).squeeze(1)

    test_loss = criterion(y_pred, y_test).item()

    # Convert logits to binary predictions
    y_pred = (torch.sigmoid(y_pred) >= 0.5).int()

    y_test = y_test.int()

    # Metrics
    sen = sensitivity(y_pred, y_test)
    spe = specificity(y_pred, y_test)
    sp  = sum_product(y_pred, y_test)

    accuracy = (y_pred == y_test).float().mean().item()

print("\n========== TEST RESULTS ==========")
print(f"Test Loss:    {test_loss:.4f}")
print(f"Accuracy:     {accuracy:.4f}")
print(f"Sensitivity:  {sen:.4f}")
print(f"Specificity:  {spe:.4f}")
print(f"Sum Product:  {sp:.4f}")
print("=================================")