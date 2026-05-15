"""Preprocessing job: build artifacts, persist via storage, optionally trigger retrain."""

from __future__ import annotations

import io
import json
import uuid

import httpx
import numpy as np

from shared.artifacts import (
    data_npz_key,
    manifest_key,
    pca_key,
    scaler_key,
)
from shared.config import PreprocessingSettings, TrainingHyperparameters
from shared.schemas.jobs import PreprocessJobResponse, RetrainJobRequest
from shared.schemas.manifests import PreprocessManifest
from shared.core.face_data import build_preprocessed_arrays, pickle_preprocessors
from shared.storage.base import StorageBackend


def run_preprocess_job(
    storage: StorageBackend,
    settings: PreprocessingSettings,
    hparams: TrainingHyperparameters,
) -> PreprocessJobResponse:
    job_id = str(uuid.uuid4())
    pd = settings.preprocessed_dir

    arrays = build_preprocessed_arrays(hparams)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        X_train=arrays.X_train,
        y_train=arrays.y_train,
        X_test=arrays.X_test,
        y_test=arrays.y_test,
    )
    buf.seek(0)
    data_ref = storage.save_bytes(data_npz_key(pd, job_id), buf.read(), content_type="application/octet-stream")

    pca_bytes, scaler_bytes = pickle_preprocessors(arrays.pca, arrays.scaler)
    pca_ref = storage.save_bytes(pca_key(pd, job_id), pca_bytes)
    scaler_ref = storage.save_bytes(scaler_key(pd, job_id), scaler_bytes)

    manifest_uri = storage.ref_from_key(manifest_key(pd, job_id)).uri
    manifest = PreprocessManifest(
        job_id=job_id,
        n_classes=len(arrays.class_names),
        input_dim_raw=int(arrays.pca.n_features_in_),
        n_pca_components=int(arrays.pca.n_components_),
        train_size=int(arrays.X_train.shape[0]),
        test_size=int(arrays.X_test.shape[0]),
        class_names=list(arrays.class_names),
        manifest_uri=manifest_uri,
        data_uri=data_ref.uri,
        pca_uri=pca_ref.uri,
        scaler_uri=scaler_ref.uri,
    )
    manifest_bytes = manifest.model_dump_json(indent=2).encode("utf-8")
    storage.save_bytes(manifest_key(pd, job_id), manifest_bytes, content_type="application/json")

    retrain_triggered = False
    if settings.trigger_retrain:
        payload = RetrainJobRequest(
            job_id=job_id,
            dataset_manifest_uri=manifest_uri,
            manifest=manifest,
        )
        url = settings.retrain_base_url.rstrip("/") + "/jobs/retrain"
        try:
            with httpx.Client(timeout=3600.0) as client:
                r = client.post(url, json=payload.model_dump())
                r.raise_for_status()
            retrain_triggered = True
        except httpx.HTTPError as exc:
            # Control-plane failure does not roll back stored artifacts (replay retrain with URI).
            raise RuntimeError(f"Retrain trigger failed: {exc}") from exc

    return PreprocessJobResponse(
        job_id=job_id,
        status="completed",
        dataset_manifest_uri=manifest_uri,
        manifest=manifest,
        retrain_triggered=retrain_triggered,
    )
