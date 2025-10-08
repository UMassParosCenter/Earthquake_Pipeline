from dataclasses import dataclass
import numpy as np
import pickle
import torch

from torch.utils.data import Dataset
from numpy.typing import NDArray

@dataclass
class Reference():
    mean: NDArray
    std: NDArray
    def normalize(self, X: NDArray) -> NDArray:
        return (X - self.mean) / self.std

class PSD_Dataset(Dataset):
    def __init__(self, X, y):
        # X shape: (events, windows, freq_bins)
        # Add channel dim for CNN2d: (events, 1, windows, freq_bins)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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


def extract_psd_array(psd_struct: dict, num_windows=11) -> NDArray:
    data = []

    # Filter keys that correspond to events
    event_keys = [k for k in psd_struct.keys() if k.startswith("event_")]

    for key in event_keys:
        event = psd_struct[key]

        # Filter and sort window keys
        window_keys = [k for k in event.keys() if k.startswith("window_")]
        window_keys = sorted(window_keys, key=lambda x: int(x.split("_")[1]))[
            :num_windows
        ]

        event_data = []
        for wk in window_keys:
            try:
                power = np.array(event[wk]["power"])
                if power.ndim == 2:
                    power = power[0, :]  # Take first channel if 2D
                event_data.append(power)
            except (KeyError, TypeError):
                continue

        if len(event_data) == num_windows:
            data.append(np.stack(event_data))  # (windows, freq_bins)

    return np.stack(data) if data else np.array([])  # (events, windows, freq_bins)


def load_pickle_data(path: str) -> dict:
    """
    Loads a pickle file and returns the 'psdResults' field if present,
    similar to how loadmat(...)[‘psdResults’] works.

    Args:
        path (str): Path to the .pkl file

    Returns:
        dict: Parsed PSD data (e.g., events with windows and power values)
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # extract 'psdResults' field if it exists
    if isinstance(data, dict) and "psdResults" in data:
        return data["psdResults"]
    return data
