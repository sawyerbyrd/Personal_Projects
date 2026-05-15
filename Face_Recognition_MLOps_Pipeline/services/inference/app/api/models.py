from fastapi import APIRouter, Depends, HTTPException

from shared.artifacts import bundle_manifest_key
from shared.config import InferenceSettings
from shared.schemas.jobs import ReloadResponse
from shared.storage import get_storage
from shared.storage.base import StorageBackend

from app.deps import get_inference_settings
from app.state import reload_predictor

router = APIRouter(tags=["models"])


def storage_dep(settings: InferenceSettings = Depends(get_inference_settings)) -> StorageBackend:
    return get_storage(settings)


@router.post("/models/reload", response_model=ReloadResponse)
def reload_model(
    settings: InferenceSettings = Depends(get_inference_settings),
    storage: StorageBackend = Depends(storage_dep),
) -> ReloadResponse:
    try:
        reload_predictor(storage, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    manifest_uri = storage.ref_from_key(bundle_manifest_key(settings.model_dir)).uri
    return ReloadResponse(status="reloaded", bundle_manifest_uri=manifest_uri)
