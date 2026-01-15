import pickle
from typing import Dict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import WeightedRandomSampler

from pipeline.cnn_utils import Spectrogram_Dataset, SpectrogramCNN
from scripts import constants
from scripts.constants import (
    BACKGROUND_DATA_PKL,
    EARTHQUAKE_DATA_PKL,
    MODEL_PTH_PATH,
)

# convert to 3D numpy arrays: (events, windows, freq_bins)
# eq_array = extract_psd_array(load_pickle_data(EARTHQUAKE_DATA_PKL))
# bg_array = extract_psd_array(load_pickle_data(BACKGROUND_DATA_PKL))

with open(EARTHQUAKE_DATA_PKL, "rb") as f:
    eq_dict: Dict = pickle.load(f)
with open(BACKGROUND_DATA_PKL, "rb") as f:
    bg_dict: Dict = pickle.load(f)

eq_array = list(eq_dict.values())
bg_array = list(bg_dict.values())

print(f'Loaded {len(eq_array)} EQ samples and {len(bg_array)} BG samples')

spectrograms = [np.asarray(s, dtype=np.float32) for s in eq_array]

X = np.concatenate([eq_array, bg_array], axis=0)
y = np.concatenate(
    [np.ones(len(eq_array), dtype=int), np.zeros(len(bg_array), dtype=int)]
)

train_idx, val_idx = train_test_split(
    np.arange(len(X)), test_size=0.2, shuffle=True, stratify=y
)

dataset = Spectrogram_Dataset(X, y)
train_set = Subset(dataset, train_idx)
val_set = Subset(dataset, val_idx)

train_labels = y[train_idx]
class_n = np.bincount(train_labels)
class_weights = 1.0 / class_n
sample_weights = class_weights[train_labels]

sampler = WeightedRandomSampler(
  weights=sample_weights,
  num_samples=len(sample_weights),
  replacement=True
)

train_loader = DataLoader(train_set, batch_size=128, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectrogramCNN().to(device)

class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), constants.RADAM_TRAINING_RATE)

for epoch in range(1, constants.N_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = X.size(0)
        running_loss += loss.item() * bs  # sum(loss * batch_size)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += bs

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            bs = X.size(0)
            running_loss += loss.item() * bs
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += bs

    val_loss = running_loss / total
    val_acc = correct / total

    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
    )

torch.save(model.state_dict(), MODEL_PTH_PATH)
print(f"Training complete. Model saved to {MODEL_PTH_PATH}")
