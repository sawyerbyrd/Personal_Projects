"""Retrain from preprocessed artifact references; MLflow + storage bundle for inference."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import httpx
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from shared.artifacts import (
    bundle_manifest_key,
    bundle_pca_key,
    bundle_scaler_key,
    model_meta_key,
    model_weights_key,
)
from shared.config import RetrainingSettings, TrainingHyperparameters
from shared.core.face_data import unpickle_preprocessors
from shared.core.face_dataset import FaceDataset
from shared.core.mlp import FaceRecognitionMLP
from shared.core.training import evaluate, get_production_f1, promote_model, train_one_epoch
from shared.schemas.jobs import RetrainJobRequest, RetrainJobResponse
from shared.schemas.manifests import ModelBundleManifest, PreprocessManifest
from shared.storage.base import StorageBackend


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_preprocess_manifest(storage: StorageBackend, manifest_uri: str) -> PreprocessManifest:
    ref = storage.resolve_uri(manifest_uri)
    raw = storage.load_file(ref)
    return PreprocessManifest.model_validate_json(raw.decode("utf-8"))


def _load_training_arrays(storage: StorageBackend, manifest: PreprocessManifest) -> tuple[FaceDataset, FaceDataset]:
    data_ref = storage.resolve_uri(manifest.data_uri)
    buf = io.BytesIO(storage.load_file(data_ref))
    z = np.load(buf)
    train_ds = FaceDataset(z["X_train"], z["y_train"])
    test_ds = FaceDataset(z["X_test"], z["y_test"])
    return train_ds, test_ds


def run_retrain_job(
    storage: StorageBackend,
    settings: RetrainingSettings,
    hparams: TrainingHyperparameters,
    body: RetrainJobRequest,
) -> RetrainJobResponse:
    manifest = body.manifest or _load_preprocess_manifest(storage, body.dataset_manifest_uri)
    if manifest.job_id != body.job_id:
        return RetrainJobResponse(
            job_id=body.job_id,
            status="failed",
            message="job_id does not match manifest.job_id",
        )

    train_ds, test_ds = _load_training_arrays(storage, manifest)
    pca_ref = storage.resolve_uri(manifest.pca_uri)
    scaler_ref = storage.resolve_uri(manifest.scaler_uri)
    pca_bytes = storage.load_file(pca_ref)
    scaler_bytes = storage.load_file(scaler_ref)
    _, _ = unpickle_preprocessors(pca_bytes, scaler_bytes)

    n_classes = manifest.n_classes
    input_dim = int(train_ds.X.shape[1])

    train_loader = DataLoader(
        train_ds,
        batch_size=hparams.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=0,
    )

    mlflow.set_tracking_uri(settings.resolved_mlflow_tracking_uri())
    mlflow.set_experiment(settings.experiment_name)
    client = MlflowClient()

    model_bundle_manifest_uri: str | None = None
    mlflow_run_id: str | None = None

    with mlflow.start_run() as run:
        mlflow_run_id = run.info.run_id

        mlflow.log_params(
            {
                "min_faces_per_person": hparams.min_faces_per_person,
                "image_resize": hparams.image_resize,
                "n_pca_components": input_dim,
                "hidden_dims": str(hparams.hidden_dims),
                "dropout": hparams.dropout,
                "epochs": hparams.epochs,
                "batch_size": hparams.batch_size,
                "learning_rate": hparams.learning_rate,
                "weight_decay": hparams.weight_decay,
                "n_classes": n_classes,
                "train_size": len(train_ds),
                "test_size": len(test_ds),
                "preprocess_job_id": manifest.job_id,
            }
        )

        model = FaceRecognitionMLP(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dims=hparams.hidden_dims,
            dropout=hparams.dropout,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams.learning_rate,
            weight_decay=hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams.epochs)

        best_val_f1 = -1.0
        best_state: dict[str, torch.Tensor] | None = None
        patience_counter = 0

        for epoch in range(1, hparams.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_f1, val_roc_auc, _, _ = evaluate(model, test_loader, criterion, DEVICE)
            scheduler.step()
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1_weighted": val_f1,
                    "val_roc_auc": val_roc_auc,
                    "lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= hparams.patience:
                    break

        assert best_state is not None
        model.load_state_dict(best_state)
        _, test_acc, test_f1, test_roc_auc, y_true, y_pred = evaluate(model, test_loader, criterion, DEVICE)

        report = classification_report(
            y_true,
            y_pred,
            target_names=manifest.class_names,
            zero_division=0,
        )
        mlflow.log_metrics(
            {
                "test_accuracy": float(test_acc),
                "test_f1_weighted": float(test_f1),
                "test_roc_auc": float(test_roc_auc),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pca_path = Path(tmpdir) / "pca.pkl"
            scaler_path = Path(tmpdir) / "scaler.pkl"
            pca_path.write_bytes(pca_bytes)
            scaler_path.write_bytes(scaler_bytes)
            mlflow.log_artifact(str(pca_path), artifact_path="preprocessors")
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessors")
            report_path = Path(tmpdir) / "classification_report.txt"
            report_path.write_text(report)
            mlflow.log_artifact(str(report_path))

        model_info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=settings.model_name,
            metadata={
                "class_names": [str(c) for c in manifest.class_names],
                "input_dim": int(input_dim),
                "n_classes": int(n_classes),
                "hidden_dims": [int(h) for h in hparams.hidden_dims],
                "dropout": float(hparams.dropout),
            },
        )
        reg_ver = getattr(model_info, "registered_model_version", None)
        new_version_str: str | None
        if reg_ver is None:
            new_version_str = None
        elif hasattr(reg_ver, "version"):
            new_version_str = str(reg_ver.version)
        else:
            new_version_str = str(reg_ver)

        if new_version_str:
            prod_f1 = get_production_f1(client, settings.model_name)
            improvement = float(test_f1) - prod_f1
            if improvement > hparams.promotion_f1_threshold:
                promote_model(client, settings.model_name, new_version_str)
                mlflow.set_tag("promoted", "true")
            else:
                mlflow.set_tag("promoted", "false")
                try:
                    client.set_registered_model_alias(
                        settings.model_name,
                        f"staging-v{new_version_str}",
                        new_version_str,
                    )
                except MlflowException:
                    pass
        else:
            mlflow.set_tag("promoted", "skipped_no_registry_version")

        md = settings.model_dir
        weights_buf = io.BytesIO()
        torch.save({"state_dict": model.state_dict()}, weights_buf)
        weights_buf.seek(0)
        w_ref = storage.save_bytes(model_weights_key(md), weights_buf.read())

        meta = {
            "input_dim": input_dim,
            "n_classes": n_classes,
            "hidden_dims": list(hparams.hidden_dims),
            "dropout": hparams.dropout,
            "class_names": manifest.class_names,
        }
        meta_ref = storage.save_bytes(
            model_meta_key(md),
            json.dumps(meta, indent=2).encode("utf-8"),
            content_type="application/json",
        )

        # Duplicate preprocessors next to weights so inference VMs can mount only MODEL_DIR.
        pca_bundle_ref = storage.save_bytes(bundle_pca_key(md), pca_bytes)
        scaler_bundle_ref = storage.save_bytes(bundle_scaler_key(md), scaler_bytes)

        bundle = ModelBundleManifest(
            job_id=manifest.job_id,
            model_weights_uri=w_ref.uri,
            meta_uri=meta_ref.uri,
            pca_uri=pca_bundle_ref.uri,
            scaler_uri=scaler_bundle_ref.uri,
            mlflow_run_id=mlflow_run_id,
            metrics={
                "test_accuracy": float(test_acc),
                "test_f1_weighted": float(test_f1),
                "test_roc_auc": float(test_roc_auc),
            },
        )
        bundle_bytes = bundle.model_dump_json(indent=2).encode("utf-8")
        bm_ref = storage.save_bytes(bundle_manifest_key(md), bundle_bytes, content_type="application/json")
        model_bundle_manifest_uri = bm_ref.uri

    if settings.notify_inference_reload:
        url = settings.inference_base_url.rstrip("/") + "/models/reload"
        try:
            with httpx.Client(timeout=60.0) as client_http:
                client_http.post(url, json={})
        except httpx.HTTPError:
            # Non-fatal: inference may be temporarily unreachable.
            pass

    return RetrainJobResponse(
        job_id=body.job_id,
        status="completed",
        model_bundle_manifest_uri=model_bundle_manifest_uri,
        message="ok",
    )
