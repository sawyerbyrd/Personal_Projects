"""PyTorch Dataset wrapper — kept separate so preprocessing doesn't require torch."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    """Thin wrapper so DataLoader can iterate over (feature, label) pairs."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
