"""LFW loading, PCA, scaler — no torch dependency (see face_dataset.py for FaceDataset)."""

from __future__ import annotations

import io
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared.config.training import TrainingHyperparameters


@dataclass
class PreprocessedArrays:
    """In-memory tensors and fitted preprocessors for persistence."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    pca: PCA
    scaler: StandardScaler
    class_names: list[str]


def build_preprocessed_arrays(hparams: TrainingHyperparameters) -> PreprocessedArrays:
    """Fetch LFW, stratified split, fit scaler+PCA on train, transform both splits."""
    lfw = fetch_lfw_people(
        min_faces_per_person=hparams.min_faces_per_person,
        resize=hparams.image_resize,
        color=False,
    )
    X_raw = lfw.data
    y = lfw.target
    class_names = list(lfw.target_names)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=hparams.test_size,
        stratify=y,
        random_state=hparams.random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    n_components = min(
        hparams.n_components,
        X_train_scaled.shape[0],
        X_train_scaled.shape[1],
    )
    pca = PCA(n_components=n_components, whiten=True, random_state=hparams.random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return PreprocessedArrays(
        X_train=X_train_pca.astype(np.float32),
        y_train=y_train.astype(np.int64),
        X_test=X_test_pca.astype(np.float32),
        y_test=y_test.astype(np.int64),
        pca=pca,
        scaler=scaler,
        class_names=class_names,
    )


def pickle_preprocessors(pca: PCA, scaler: StandardScaler) -> tuple[bytes, bytes]:
    """Serialize preprocessors to bytes for storage.save_bytes (no direct Path usage)."""
    buf_pca = io.BytesIO()
    pickle.dump(pca, buf_pca)
    buf_scaler = io.BytesIO()
    pickle.dump(scaler, buf_scaler)
    return buf_pca.getvalue(), buf_scaler.getvalue()


def unpickle_preprocessors(pca_bytes: bytes, scaler_bytes: bytes) -> tuple[PCA, StandardScaler]:
    pca = pickle.loads(pca_bytes)
    scaler = pickle.loads(scaler_bytes)
    return pca, scaler
