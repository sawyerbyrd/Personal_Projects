"""MLP classifier — adapted from the original ``model/model.py``."""

from __future__ import annotations

import torch
import torch.nn as nn


class FaceRecognitionMLP(nn.Module):
    """Multi-layer perceptron for closed-set face identification on PCA features."""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
