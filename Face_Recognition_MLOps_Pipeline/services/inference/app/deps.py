from functools import lru_cache

from shared.config import InferenceSettings


@lru_cache
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()
