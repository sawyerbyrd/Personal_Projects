"""Process-wide predictor handle (safe reload under a lock)."""

from __future__ import annotations

from threading import RLock

from shared.config import InferenceSettings
from shared.storage.base import StorageBackend

from app.services.predictor import FacePredictorStorage

_lock = RLock()
_predictor: FacePredictorStorage | None = None


def get_or_create_predictor(storage: StorageBackend, settings: InferenceSettings) -> FacePredictorStorage:
    global _predictor
    with _lock:
        if _predictor is None:
            _predictor = FacePredictorStorage(storage, settings)
        return _predictor


def reload_predictor(storage: StorageBackend, settings: InferenceSettings) -> FacePredictorStorage:
    global _predictor
    with _lock:
        _predictor = FacePredictorStorage(storage, settings)
        return _predictor
