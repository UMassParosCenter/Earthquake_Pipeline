import pickle
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import WeightedRandomSampler

from pipeline.cnn_utils import EarlyStopping, Spectrogram_Dataset, SpectrogramCNN
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

eq_spectrograms = [item[0] for item in eq_dict.values()]  # Spectrograms
eq_power = np.array([item[1] for item in eq_dict.values()])  # Power features

bg_spectrograms = [item[0] for item in bg_dict.values()]
bg_power = np.array([item[1] for item in bg_dict.values()])

print(f'Loaded {len(eq_spectrograms)} EQ samples and {len(bg_spectrograms)} BG samples')
print(f'Spectrogram shape: {eq_spectrograms[0].shape}')
print(f'Power features shape: {eq_power[0].shape}')

X_spec = np.array(eq_spectrograms + bg_spectrograms, dtype=np.float32)
X_power = np.vstack([eq_power, bg_power])
y = np.concatenate(
    [np.ones(len(eq_spectrograms), dtype=int), np.zeros(len(bg_spectrograms), dtype=int)]
)

train_idx, val_idx = train_test_split(
    np.arange(len(X_spec)), test_size=0.2, shuffle=True, stratify=y, random_state=42
)

dataset = Spectrogram_Dataset(X_spec, X_power, y)
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

train_loader = DataLoader(train_set, batch_size=512, sampler=sampler)
val_loader = DataLoader(val_set, batch_size=512, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpectrogramCNN().to(device)

class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

criterion = CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), constants.RADAM_TRAINING_RATE)

early_stopping = EarlyStopping(
    patience=constants.EARLY_STOPPING_PATIENCE, min_delta=constants.EARLY_STOPPING_MIN_DELTA
)

best_score = None

for epoch in range(1, constants.N_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for spec_batch, power_batch, label_batch in train_loader:
      spec_batch = spec_batch.to(device)
      power_batch = power_batch.to(device)
      label_batch = label_batch.to(device)

      optimizer.zero_grad()
      logits = model(spec_batch, power_batch)
      loss = criterion(logits, label_batch)
      loss.backward()
      optimizer.step()

      bs = spec_batch.size(0)
      running_loss += loss.item() * bs  # sum(loss * batch_size)
      correct += (logits.argmax(dim=1) == label_batch).sum().item()
      total += bs

    train_loss = running_loss / total
    train_acc = correct / total

    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
      for spec_batch, power_batch, label_batch in val_loader:
        spec_batch = spec_batch.to(device)
        power_batch = power_batch.to(device)
        label_batch = label_batch.to(device)

        logits = model(spec_batch, power_batch)
        loss = criterion(logits, label_batch)

        bs = spec_batch.size(0)
        running_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == label_batch).sum().item()
        total += bs

    val_loss = running_loss / total
    val_acc = correct / total

    early_stopping(val_loss)
    best = False
    if best_score is None or val_loss < best_score:
      torch.save(model.state_dict(), MODEL_PTH_PATH)
      best_score = val_loss
      best = True


    print(
        f"Epoch {epoch:02d} | "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}" +
        (" | Best score, saved" if best else "")
    )

model.eval()
running_loss = 0.0
all_preds = []
all_labels = []
all_probs = []

test_loader = DataLoader(dataset, batch_size=2056, shuffle=False)

with torch.no_grad():
  for spec_batch, power_batch, label_batch in test_loader:
      spec_batch = spec_batch.to(device)
      power_batch = power_batch.to(device)

      logits = model(spec_batch, power_batch)
      probs = torch.softmax(logits, dim=1)
      preds = logits.argmax(dim=1)

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(label_batch.cpu().numpy())
      all_probs.extend(probs[:, 1].cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calculate metrics
avg_loss = running_loss / len(all_labels)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, pos_label=1)
recall = recall_score(all_labels, all_preds, pos_label=1)
# f1 = f1_score(all_labels, all_preds, pos_label=1)
cm = confusion_matrix(all_labels, all_preds)

print(f"{'='*60}")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
# print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix (rows: true, cols: predicted) [0=Background, 1=Earthquake]:")
print(cm)

print(f"Training complete. Model saved to {MODEL_PTH_PATH}")
