from fastapi import APIRouter, Depends

from shared.config import PreprocessingSettings, TrainingHyperparameters
from shared.schemas.jobs import PreprocessJobResponse
from shared.storage import get_storage
from shared.storage.base import StorageBackend

from app.deps import get_preprocessing_settings, get_training_hparams
from app.services.preprocess_job import run_preprocess_job

router = APIRouter(tags=["jobs"])


def storage_dep(settings: PreprocessingSettings = Depends(get_preprocessing_settings)) -> StorageBackend:
    return get_storage(settings)


@router.post("/jobs/preprocess", response_model=PreprocessJobResponse)
def preprocess_job(
    settings: PreprocessingSettings = Depends(get_preprocessing_settings),
    hparams: TrainingHyperparameters = Depends(get_training_hparams),
    storage: StorageBackend = Depends(storage_dep),
) -> PreprocessJobResponse:
    """Run LFW fetch + PCA pipeline and persist artifacts; optionally trigger retraining."""
    return run_preprocess_job(storage, settings, hparams)
