"""Cross-service job and inference API DTOs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from shared.schemas.manifests import PreprocessManifest


class RetrainJobRequest(BaseModel):
    """Control-plane payload: references only."""

    job_id: str
    dataset_manifest_uri: str
    manifest: PreprocessManifest | None = Field(
        default=None,
        description="Optional inline manifest; authoritative copy is at dataset_manifest_uri.",
    )


class RetrainJobResponse(BaseModel):
    job_id: str
    status: str
    model_bundle_manifest_uri: str | None = None
    message: str | None = None


class PreprocessJobResponse(BaseModel):
    job_id: str
    status: str
    dataset_manifest_uri: str
    manifest: PreprocessManifest
    retrain_triggered: bool = False


class PredictRequest(BaseModel):
    """Single-vector prediction (already PCA-transformed)."""

    features: list[float] = Field(
        ...,
        description="PCA-transformed feature vector (length = n_pca_components).",
    )


class PredictImageRequest(BaseModel):
    """Raw pixel prediction — server applies scaler + PCA before inference."""

    pixels: list[float] = Field(
        ...,
        description="Flattened raw pixel values from lfw.data (same resize as training).",
    )


class PredictResponse(BaseModel):
    class_index: int
    class_name: str | None = None
    confidence: float
    model_version_label: str


class ReloadResponse(BaseModel):
    status: str
    bundle_manifest_uri: str | None = None
