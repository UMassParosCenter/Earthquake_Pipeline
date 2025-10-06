"""
EarthquakeCNN2d Model Definition for Infrasound PSD Classification
------------------------------------------------------------------

This script defines a 2D Convolutional Neural Network architecture tailored
for classifying earthquake vs background events using Power Spectral Density
(PSD) features extracted from infrasound waveform data.

Key Components:
---------------
- ConvBlock2d: A reusable convolutional block including convolution, batch
  normalization, ReLU activation, max pooling, and dropout for regularization.
- EarthquakeCNN2d: The main CNN model consisting of two ConvBlock2d layers,
  followed by fully connected layers that output class logits for binary
  classification (earthquake or background).

Features:
---------
- Automatic padding calculation to preserve spatial dimensions during convolution.
- Dynamic computation of the flattened feature vector size to accommodate varying
  input PSD dimensions (windows x frequency bins).
- Uses ReLU activations and dropout for effective training and generalization.
- Outputs raw logits for subsequent use with softmax and cross-entropy loss.

Inputs and Outputs:
-------------------
- Input: 4D tensor with shape (batch_size, 1, windows, freq_bins), representing PSD
  features as a 2D image with one channel.
- Output: Tensor with shape (batch_size, 2) containing logits for the two classes.

Author: Ethan Gelfand
Date: 08/12/2025
"""

import torch
import torch.nn as nn

# --- Model Definition ---
class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel=2, padding=None):
        super().__init__()
        
        # Automatically compute padding if not given
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class EarthquakeCNN2d(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # input_shape: (windows, freq_bins)
        self.conv1 = ConvBlock2d(1, 16, kernel_size=(5, 5))
        self.conv2 = ConvBlock2d(16, 32, kernel_size=(3, 5))  # more time context
        
        # Calculate flattened feature size dynamically
        self.flatten_dim = self._get_flattened_size(input_shape)

        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc_hidden = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 2)

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_shape)
            x = self.conv1(dummy)
            x = self.conv2(x)
            return x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc_hidden(x))
        x = self.fc2(x)
        return x