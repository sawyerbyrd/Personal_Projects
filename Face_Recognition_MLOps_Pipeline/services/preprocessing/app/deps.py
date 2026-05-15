from functools import lru_cache

from shared.config import PreprocessingSettings, TrainingHyperparameters


@lru_cache
def get_preprocessing_settings() -> PreprocessingSettings:
    return PreprocessingSettings()


@lru_cache
def get_training_hparams() -> TrainingHyperparameters:
    return TrainingHyperparameters()
