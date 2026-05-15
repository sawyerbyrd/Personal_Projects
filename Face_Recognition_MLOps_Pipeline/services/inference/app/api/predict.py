from fastapi import APIRouter, Depends, HTTPException

from shared.config import InferenceSettings
from shared.schemas.jobs import PredictImageRequest, PredictRequest, PredictResponse
from shared.storage import get_storage
from shared.storage.base import StorageBackend

from app.deps import get_inference_settings
from app.state import get_or_create_predictor

router = APIRouter(tags=["predict"])


def storage_dep(settings: InferenceSettings = Depends(get_inference_settings)) -> StorageBackend:
    return get_storage(settings)


def _make_response(predictor, class_idx: int, conf: float) -> PredictResponse:
    name = predictor.class_names[class_idx] if predictor.class_names else None
    return PredictResponse(
        class_index=class_idx,
        class_name=name,
        confidence=conf,
        model_version_label=predictor.version_label,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest,
    settings: InferenceSettings = Depends(get_inference_settings),
    storage: StorageBackend = Depends(storage_dep),
) -> PredictResponse:
    try:
        predictor = get_or_create_predictor(storage, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    idx, conf = predictor.predict_vector(body.features)
    return _make_response(predictor, idx, conf)


@router.post("/predict/image", response_model=PredictResponse)
def predict_image(
    body: PredictImageRequest,
    settings: InferenceSettings = Depends(get_inference_settings),
    storage: StorageBackend = Depends(storage_dep),
) -> PredictResponse:
    try:
        predictor = get_or_create_predictor(storage, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    idx, conf = predictor.predict_raw(body.pixels)
    return _make_response(predictor, idx, conf)
