from functools import lru_cache

from shared.config import RetrainingSettings, TrainingHyperparameters


@lru_cache
def get_retraining_settings() -> RetrainingSettings:
    return RetrainingSettings()


@lru_cache
def get_training_hparams() -> TrainingHyperparameters:
    return TrainingHyperparameters()
