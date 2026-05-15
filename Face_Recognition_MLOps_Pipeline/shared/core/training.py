"""Training / evaluation helpers — adapted from the original ``model/train.py``."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * len(y_batch)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total
    all_probs_np = np.array(all_probs)
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    f1 = f1_score(all_labels_np, all_preds_np, average="weighted", zero_division=0)
    try:
        roc_auc = roc_auc_score(
            all_labels_np,
            all_probs_np,
            multi_class="ovr",
            average="weighted",
        )
    except ValueError:
        roc_auc = float("nan")
    return avg_loss, accuracy, f1, roc_auc, all_labels_np, all_preds_np


def get_production_f1(client: MlflowClient, model_name: str) -> float:
    try:
        version = client.get_model_version_by_alias(model_name, "production")
        run = client.get_run(version.run_id)
        return float(run.data.metrics.get("test_f1_weighted", -1.0))
    except MlflowException:
        return -1.0


def promote_model(client: MlflowClient, model_name: str, version: str) -> None:
    try:
        client.delete_registered_model_alias(model_name, "production")
    except MlflowException:
        pass
    client.set_registered_model_alias(model_name, "production", version)
