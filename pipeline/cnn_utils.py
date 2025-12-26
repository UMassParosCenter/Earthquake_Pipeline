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
    def __init__(self) -> None:
        super().__init__()

        self.features = Sequential(
            nn.Conv2d(1, 18, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(18, 36, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(36, 54, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(54, 54, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
        )
        test = torch.zeros(1, 1, 49, 39)
        with torch.no_grad():
            test = self.features(test)
            test = test.view(test.size(0), -1)
        self.fc = nn.Linear(test.shape[1], 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
