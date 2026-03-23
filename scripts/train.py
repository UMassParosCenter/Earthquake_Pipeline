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
from sklearn.model_selection import StratifiedKFold
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from pipeline.cnn_utils import EarlyStopping, Spectrogram_Dataset, SpectrogramCNN
from scripts import constants
from scripts.constants import (
    BACKGROUND_DATA_PKL,
    EARTHQUAKE_DATA_PKL,
    MODEL_PTH_PATH,
)

# Load data and combine power and spectrograms

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
dataset = Spectrogram_Dataset(X_spec, X_power, y)

n_splits = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Store metrics for each fold
fold_metrics = {
    'precision': [],
    'recall': [],
    'f1': [],
    'val_loss': []
}
best_fold_score = 0

fold = 1

for train_idx, val_idx in skf.split(X_power, y):
  print(f"Fold {fold}/{n_splits}")
  print(f"{'='*20}")

  train_set = Subset(dataset, train_idx.tolist())
  val_set = Subset(dataset, val_idx.tolist())

  train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

  model = SpectrogramCNN().to(device)

  criterion = CrossEntropyLoss(weight=torch.tensor([1, 1.5], dtype=torch.float32).to(device))
  optimizer = torch.optim.Adam(model.parameters(), constants.RADAM_TRAINING_RATE)

  early_stopping = EarlyStopping(
      patience=constants.EARLY_STOPPING_PATIENCE, min_delta=constants.EARLY_STOPPING_MIN_DELTA
  )
  best_val_loss = float('inf')
  best_model_state = None

  for epoch in range(1, constants.N_EPOCHS):
    model.train()
    train_loss = 0.0
    for specs, powers, labels in train_loader:
      specs = specs.to(device)
      powers = powers.to(device)
      labels = labels.to(device)

      optimizer.zero_grad()
      outputs = model(specs, powers)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
      for specs, powers, labels in val_loader:
        specs = specs.to(device)
        powers = powers.to(device)
        labels = labels.to(device)

        outputs = model(specs, powers)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    save = val_loss < best_val_loss
    if save:
      best_val_loss = val_loss
      best_model_state = model.state_dict()

    if (epoch - 1) % 10 == 0 or epoch == 1 or save:
      precision = precision_score(all_labels, all_preds, zero_division=0.0)
      recall = recall_score(all_labels, all_preds, zero_division=0.0)
      f1 = f1_score(all_labels, all_preds, zero_division=0.0)
      print(f"Epoch {epoch:02d} | Val Loss: {val_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}" + (" | Saved" if save else ""))

  # Metrics for this fold
  assert best_model_state is not None
  model.load_state_dict(best_model_state)
  model.eval()
  all_preds = []
  all_labels = []

  with torch.no_grad():
    for specs, powers, labels in val_loader:
        specs = specs.to(device)
        powers = powers.to(device)

        outputs = model(specs, powers)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

  precision = precision_score(all_labels, all_preds, zero_division=0.0)
  recall = recall_score(all_labels, all_preds, zero_division=0.0)
  f1 = f1_score(all_labels, all_preds, zero_division=0.0)

  fold_metrics['precision'].append(precision)
  fold_metrics['recall'].append(recall)
  fold_metrics['f1'].append(f1)

  print(f"\nFold {fold} Results:")
  print(f"  Precision: {precision:.4f}")
  print(f"  Recall: {recall:.4f}")
  print(f"  F1: {f1:.4f}")

  if f1 > best_fold_score:
    best_fold_score = f1
    torch.save(model.state_dict(), MODEL_PTH_PATH)

  fold += 1

print("\nFINAL CROSS-VALIDATION RESULTS")
print(f"{'='*20}")
print(f"Precision: {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
print(f"Recall:    {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
print(f"F1-Score:  {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")

model = SpectrogramCNN().to(device)
model.load_state_dict(torch.load(MODEL_PTH_PATH, map_location="cpu"))
model.eval()
val_loss = 0.0
all_preds = []
all_labels = []
all_probs = []

test_loader = DataLoader(dataset, batch_size=2056, shuffle=False)

with torch.no_grad():
  for specs, powers, labels in test_loader:
      specs = specs.to(device)
      powers = powers.to(device)

      outputs = model(specs, powers)
      probs = torch.softmax(outputs, dim=1)
      preds = outputs.argmax(dim=1)

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
      all_probs.extend(probs[:, 1].cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Calculate metrics
avg_loss = val_loss / len(all_labels)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, pos_label=1)
recall = recall_score(all_labels, all_preds, pos_label=1)
f1 = f1_score(all_labels, all_preds, pos_label=1)
cm = confusion_matrix(all_labels, all_preds)

print("\nRESULTS ACROSS ENTIRE DATASET")
print(f"{'='*20}")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nConfusion Matrix (rows: true, cols: predicted) [0=Background, 1=Earthquake]:")
print(cm)

print(f"Training complete. Model saved to {MODEL_PTH_PATH}")
