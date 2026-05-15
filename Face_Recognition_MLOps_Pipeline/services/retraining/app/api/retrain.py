from fastapi import APIRouter, Depends

from shared.config import RetrainingSettings, TrainingHyperparameters
from shared.schemas.jobs import RetrainJobRequest, RetrainJobResponse
from shared.storage import get_storage
from shared.storage.base import StorageBackend

from app.deps import get_retraining_settings, get_training_hparams
from app.services.retrain_job import run_retrain_job

router = APIRouter(tags=["jobs"])


def storage_dep(settings: RetrainingSettings = Depends(get_retraining_settings)) -> StorageBackend:
    return get_storage(settings)


@router.post("/jobs/retrain", response_model=RetrainJobResponse)
def retrain_job(
    body: RetrainJobRequest,
    settings: RetrainingSettings = Depends(get_retraining_settings),
    hparams: TrainingHyperparameters = Depends(get_training_hparams),
    storage: StorageBackend = Depends(storage_dep),
) -> RetrainJobResponse:
    """Train from preprocessed artifact URIs; log MLflow; write latest inference bundle."""
    return run_retrain_job(storage, settings, hparams, body)
