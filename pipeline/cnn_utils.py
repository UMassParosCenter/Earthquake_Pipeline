import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
from torch.utils.data import Dataset


class Spectrogram_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        spec = self.X[idx]
        spec = spec.unsqueeze(0)
        label = self.y[idx]
        return spec, label

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = loss
            return
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class SpectrogramCNN(nn.Module):
  def __init__(self, dropout_rate=0.3):
    super().__init__()

    # Feature extraction (works with any input size)
    self.features = nn.Sequential(
        # Block 1
        nn.Conv2d(1, 32, kernel_size=3, padding='same'),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding='same'),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2, 2)),
        nn.Dropout2d(p=dropout_rate),

        # Block 2
        nn. Conv2d(32, 64, kernel_size=3, padding='same'),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding='same'),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2, 2)),
        nn.Dropout2d(p=dropout_rate),

        # Block 3
        nn.Conv2d(64, 128, kernel_size=3, padding='same'),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, padding='same'),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2, 2)),
        nn.Dropout2d(p=dropout_rate),

        # Block 4
        nn.Conv2d(128, 256, kernel_size=3, padding='same'),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, kernel_size=3, padding='same'),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2, 2)),
        nn.Dropout2d(p=dropout_rate),
    )

    # Global pooling (handles any spatial size)
    self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

    # Classifier (input size is always 256*2 regardless of input image size)
    self.classifier = nn.Sequential(
        nn. Linear(256 * 2, 512),  # *2 for avg + max pooling
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(128, 2)
    )

  def forward(self, x):
      # Feature extraction
      x = self.features(x)

      # Global pooling (works with any spatial size)
      avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
      max_pool = self.global_max_pool(x).view(x.size(0), -1)
      x = torch.cat([avg_pool, max_pool], dim=1)

      # Classification
      return self.classifier(x)
