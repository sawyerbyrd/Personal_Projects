"""Load latest inference bundle from storage (no MLflow dependency on inference VM)."""

from __future__ import annotations

import io
import json

import torch

from shared.artifacts import bundle_manifest_key
from shared.config import InferenceSettings
from shared.core.face_data import unpickle_preprocessors
from shared.core.mlp import FaceRecognitionMLP
from shared.schemas.manifests import ModelBundleManifest
from shared.storage.base import StorageBackend

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FacePredictorStorage:
    """Loads ``bundle_manifest.json`` + weights + preprocessors via ``StorageBackend``."""

    def __init__(self, storage: StorageBackend, settings: InferenceSettings) -> None:
        self._storage = storage
        self._settings = settings
        self.model: FaceRecognitionMLP
        self.bundle: ModelBundleManifest
        self._version_label: str
        self._load()

    def _load(self) -> None:
        md = self._settings.model_dir
        ref = self._storage.ref_from_key(bundle_manifest_key(md))
        if not self._storage.exists(ref):
            raise FileNotFoundError(
                "No inference bundle found. Run preprocessing + retraining first "
                f"(expected {bundle_manifest_key(md)} under storage root)."
            )
        raw = self._storage.load_file(ref)
        self.bundle = ModelBundleManifest.model_validate_json(raw.decode("utf-8"))

        weights_ref = self._storage.resolve_uri(self.bundle.model_weights_uri)
        blob = io.BytesIO(self._storage.load_file(weights_ref))
        try:
            ckpt = torch.load(blob, map_location=DEVICE, weights_only=False)
        except TypeError:
            blob.seek(0)
            ckpt = torch.load(blob, map_location=DEVICE)

        meta_ref = self._storage.resolve_uri(self.bundle.meta_uri)
        meta = json.loads(self._storage.load_file(meta_ref).decode("utf-8"))

        hidden = meta["hidden_dims"]
        if isinstance(hidden, list):
            hidden_dims: tuple[int, ...] = tuple(int(h) for h in hidden)
        else:
            hidden_dims = tuple(hidden)

        self.model = FaceRecognitionMLP(
            input_dim=int(meta["input_dim"]),
            n_classes=int(meta["n_classes"]),
            hidden_dims=hidden_dims,
            dropout=float(meta["dropout"]),
        ).to(DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        pca_ref = self._storage.resolve_uri(self.bundle.pca_uri)
        scaler_ref = self._storage.resolve_uri(self.bundle.scaler_uri)
        pca_bytes = self._storage.load_file(pca_ref)
        scaler_bytes = self._storage.load_file(scaler_ref)
        self.pca, self.scaler = unpickle_preprocessors(pca_bytes, scaler_bytes)
        self.class_names: list[str] = meta.get("class_names", [])

        self._version_label = f"{self.bundle.job_id}@{self.bundle.trained_at}"

    def reload(self) -> None:
        self._load()

    @torch.no_grad()
    def predict_vector(self, features: list[float]) -> tuple[int, float]:
        x = torch.tensor([features], dtype=torch.float32, device=DEVICE)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        class_idx = int(probs.argmax().item())
        confidence = float(probs[class_idx].item())
        return class_idx, confidence

    @torch.no_grad()
    def predict_raw(self, pixels: list[float]) -> tuple[int, float]:
        """Accept raw pixel values, apply scaler+PCA, then run the model."""
        import numpy as np

        arr = np.array(pixels, dtype=np.float64).reshape(1, -1)
        scaled = self.scaler.transform(arr)
        pca_vec = self.pca.transform(scaled).astype(np.float32)
        return self.predict_vector(pca_vec[0].tolist())

    @property
    def version_label(self) -> str:
        return self._version_label
