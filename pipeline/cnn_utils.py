import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Spectrogram_Dataset(Dataset):
    def __init__(self, spectrograms, power_features, labels):
        self.spectrograms = torch.tensor(spectrograms, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.power_features = torch.tensor(
            power_features,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        spec = self.spectrograms[idx].unsqueeze(0)
        power = self.power_features[idx]
        label = self.labels[idx]
        return spec, power, label


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

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 18, kernel_size=3, padding="same"),
            nn.BatchNorm2d(18),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout_rate),
            # Block 2
            nn.Conv2d(18, 36, kernel_size=3, padding="same"),
            nn.BatchNorm2d(36),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout_rate),
            # Block 3
            nn.Conv2d(36, 54, kernel_size=3, padding="same"),
            nn.BatchNorm2d(54),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout_rate),
            # Block 4
            nn.Conv2d(54, 54, kernel_size=3, padding="same"),
            nn.BatchNorm2d(54),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(p=dropout_rate),
        )

        # MLP for absolute power features
        self.power_branch = nn.Sequential(
            nn.Linear(9, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(5 * 2 * 54 + 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 2),
        )

    def forward(self, spectrogram, power_features):
        # Process log-scale spectrogram for patterns
        spec_feat = self.features(spectrogram).flatten(1)
        # Process absolute power features
        power_feat = self.power_branch(power_features)

        # Combine both information sources
        combined = torch.cat([spec_feat, power_feat], dim=1)

        # Final classification
        return self.classifier(combined)
