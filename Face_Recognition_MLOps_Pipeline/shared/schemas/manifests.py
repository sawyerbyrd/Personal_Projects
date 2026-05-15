"""Manifests persisted as JSON next to large artifacts (metadata plane)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class PreprocessManifest(BaseModel):
    """Written alongside preprocessed tensors and pickles (metadata for retraining)."""

    job_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    schema_version: str = "1.0"
    n_classes: int
    input_dim_raw: int
    n_pca_components: int
    train_size: int
    test_size: int
    class_names: list[str]
    manifest_uri: str
    data_uri: str
    pca_uri: str
    scaler_uri: str
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelBundleManifest(BaseModel):
    """Everything inference needs; written by retraining."""

    job_id: str
    trained_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    schema_version: str = "1.0"
    model_weights_uri: str
    meta_uri: str
    pca_uri: str
    scaler_uri: str
    mlflow_run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
