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


class SpectrogramCNN(nn.Module):
    def __init__(self, dropout_rate=0.3) -> None:
        super().__init__()

        self.features = Sequential(
            # First conv block
            nn.Conv2d(1, 18, kernel_size=3, padding="same"),
            nn.BatchNorm2d(18),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn. Dropout2d(p=dropout_rate),

            # Second conv block
            nn.Conv2d(18, 36, kernel_size=3, padding="same"),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout_rate),

            # Third conv block
            nn.Conv2d(36, 54, kernel_size=3, padding="same"),
            nn.BatchNorm2d(54),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout_rate),

            # Fourth conv block
            nn.Conv2d(54, 54, kernel_size=3, padding="same"),
            nn.BatchNorm2d(54),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout_rate),
        )

        # Calculate flattened feature size
        test = torch.zeros(1, 1, 49, 39)
        with torch.no_grad():
            test = self.features(test)
            test = test.view(test.size(0), -1)

        # Classifier with additional hidden layer and dropout
        self.classifier = Sequential(
            nn.Linear(test.shape[1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
