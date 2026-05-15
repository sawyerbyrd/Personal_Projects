"""Environment-driven settings. Each service imports the slice it needs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageSettings(BaseSettings):
    """
    Storage roots and logical directory names.

    TODO(infra): For mounted shared storage, set STORAGE_ROOT to the mount point
    (same path visible on all VMs) and keep STORAGE_BACKEND=local — see
    shared/storage/local.py and factories.py for MountedFilesystemStorage hook.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    storage_backend: Literal["local", "s3"] = Field(
        default="local",
        validation_alias="STORAGE_BACKEND",
        description="Backend id: 'local' or 's3'.",
    )
    s3_bucket: str = Field(default="", validation_alias="S3_BUCKET")
    s3_prefix: str = Field(default="", validation_alias="S3_PREFIX")
    s3_region: str = Field(default="us-east-1", validation_alias="S3_REGION")
    s3_endpoint_url: str = Field(default="", validation_alias="S3_ENDPOINT_URL")
    storage_root: Path = Field(
        default=Path("./storage"),
        validation_alias="STORAGE_ROOT",
    )
    preprocessed_dir: str = Field(
        default="preprocessed",
        validation_alias="PREPROCESSED_DIR",
        description="Subdirectory under storage_root for preprocessing outputs.",
    )
    model_dir: str = Field(
        default="models/latest",
        validation_alias="MODEL_DIR",
        description="Subdirectory under storage_root for the latest inference bundle.",
    )
    mlruns_dir: str = Field(
        default="mlruns",
        validation_alias="MLRUNS_DIR",
        description="Subdirectory under storage_root for MLflow file store (retraining only).",
    )

    @property
    def preprocessed_root(self) -> Path:
        return self.storage_root / self.preprocessed_dir

    @property
    def model_bundle_root(self) -> Path:
        return self.storage_root / self.model_dir

    @property
    def mlflow_file_store(self) -> Path:
        return self.storage_root / self.mlruns_dir


class PreprocessingSettings(StorageSettings):
    """Preprocessing service: outbound calls and training-related defaults for manifests."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    retrain_base_url: str = Field(
        default="http://localhost:8002",
        validation_alias="RETRAIN_BASE_URL",
        description="Base URL of the retraining FastAPI service (control plane).",
    )
    trigger_retrain: bool = Field(
        default=True,
        validation_alias="PREPROCESS_TRIGGER_RETRAIN",
        description="If true, POST /jobs/retrain after a successful preprocess job.",
    )


class RetrainingSettings(StorageSettings):
    """Retraining service: MLflow + optional inference notification."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mlflow_tracking_uri: str = Field(
        default="",
        validation_alias="MLFLOW_TRACKING_URI",
        description="MLflow tracking URI (http(s):// or file:...). "
        "If empty, uses ``file:{STORAGE_ROOT}/{MLRUNS_DIR}``.",
    )
    experiment_name: str = Field(
        default="face_recognition_lfw",
        validation_alias="MLFLOW_EXPERIMENT_NAME",
    )
    model_name: str = Field(
        default="face_recognition_model",
        validation_alias="MLFLOW_MODEL_NAME",
    )
    inference_base_url: str = Field(
        default="http://localhost:8003",
        validation_alias="INFERENCE_BASE_URL",
    )
    notify_inference_reload: bool = Field(
        default=True,
        validation_alias="RETRAIN_NOTIFY_INFERENCE",
    )

    def resolved_mlflow_tracking_uri(self) -> str:
        """Resolve MLflow file store default under ``storage_root`` / ``mlruns_dir``."""
        uri = (self.mlflow_tracking_uri or "").strip()
        if uri.startswith("http://") or uri.startswith("https://") or uri.startswith("file:"):
            return uri
        if uri:
            p = Path(uri)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            else:
                p = p.resolve()
            return f"file:{p.as_posix()}"
        self.mlflow_file_store.mkdir(parents=True, exist_ok=True)
        p = self.mlflow_file_store.resolve()
        return f"file:{p.as_posix()}"


class InferenceSettings(StorageSettings):
    """Inference service loads the latest bundle from storage."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
