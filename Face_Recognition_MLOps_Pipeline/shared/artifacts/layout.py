"""
Logical keys under ``STORAGE_ROOT`` (POSIX-style segments).

TODO(infra): S3 implementations should map these logical keys to object key prefixes
(e.g. ``{env}/{preprocessed_dir}/{job_id}/...``) in one place.
"""

from __future__ import annotations


def _norm(seg: str) -> str:
    return seg.replace("\\", "/").strip("/")


def preprocess_job_prefix(preprocessed_dir: str, job_id: str) -> str:
    return f"{_norm(preprocessed_dir)}/{job_id}"


def manifest_key(preprocessed_dir: str, job_id: str) -> str:
    return f"{preprocess_job_prefix(preprocessed_dir, job_id)}/manifest.json"


def data_npz_key(preprocessed_dir: str, job_id: str) -> str:
    return f"{preprocess_job_prefix(preprocessed_dir, job_id)}/data.npz"


def pca_key(preprocessed_dir: str, job_id: str) -> str:
    return f"{preprocess_job_prefix(preprocessed_dir, job_id)}/pca.pkl"


def scaler_key(preprocessed_dir: str, job_id: str) -> str:
    return f"{preprocess_job_prefix(preprocessed_dir, job_id)}/scaler.pkl"


def model_weights_key(model_dir: str) -> str:
    return f"{_norm(model_dir)}/model.pt"


def model_meta_key(model_dir: str) -> str:
    return f"{_norm(model_dir)}/model_meta.json"


def bundle_manifest_key(model_dir: str) -> str:
    return f"{_norm(model_dir)}/bundle_manifest.json"


def bundle_pca_key(model_dir: str) -> str:
    """Copy of preprocessors co-located with weights for inference-only storage visibility."""
    return f"{_norm(model_dir)}/pca.pkl"


def bundle_scaler_key(model_dir: str) -> str:
    return f"{_norm(model_dir)}/scaler.pkl"
