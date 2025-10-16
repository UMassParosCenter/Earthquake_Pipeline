import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch_optimizer as optim
from sklearn.utils.class_weight import compute_class_weight

from scripts.constants import (
    BACKGROUND_PSD_PKL,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EARTHQUAKE_PSD_PKL,
    MODEL_PTH_PATH,
    N_EPOCHS,
    RADAM_TRAINING_RATE,
    REFERENCE_NPZ_PATH,
    TRAINING_LOG_PATH,
)
from pipeline.dataset_utils import (
    Reference,
    load_pickle_data,
    extract_psd_array,
    PSD_Dataset,
    EarlyStopping,
)
from pipeline.cnn_model import EarthquakeCNN2d
from tqdm import tqdm
import csv

# convert to 3D numpy arrays: (events, windows, freq_bins)
eq_array = extract_psd_array(load_pickle_data(EARTHQUAKE_PSD_PKL))
bg_array = extract_psd_array(load_pickle_data(BACKGROUND_PSD_PKL))

# Assign labels
eq_labels = np.ones(len(eq_array), dtype=int)
bg_labels = np.zeros(len(bg_array), dtype=int)
X = np.concatenate([eq_array, bg_array], axis=0)
y = np.concatenate([eq_labels, bg_labels], axis=0)

# Shuffle dataset
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Apply log transform globally
X_log = np.log10(X + 1e-12)

# Compute normalization
reference = Reference(X_log.mean(axis=0), X_log.std(axis=0) + 1e-12)
X_norm = reference.normalize(X_log)

# Save normalization stats
np.savez(REFERENCE_NPZ_PATH, mean=reference.mean, std=reference.std)

# Compute class weights
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Prepare DataLoader
dataset = PSD_Dataset(X_norm, y)
train_loader = DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=min(4, os.cpu_count() or 1)
)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model, loss, optimizer, early stopping
model = EarthquakeCNN2d(input_shape=X.shape[1:]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.RAdam(model.parameters(), lr=RADAM_TRAINING_RATE)
early_stopping = EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA
)

# Train model
with open(TRAINING_LOG_PATH, mode="w", newline="") as log_file:
    writer = csv.writer(log_file)
    writer.writerow(["epoch", "loss", "accuracy"])  # header row

    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            pred = model(data)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss /= len(train_loader)
        train_acc = correct / total

        # log to CSV
        writer.writerow([epoch + 1, round(train_loss, 6), round(train_acc, 6)])

        early_stopping(train_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

# Save final model
torch.save(model.state_dict(), MODEL_PTH_PATH)
print(f"Training complete. Model saved to {MODEL_PTH_PATH}")
