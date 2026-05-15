"""Default training / preprocessing hyperparameters (tunable without changing services)."""

from pydantic import BaseModel, Field


class TrainingHyperparameters(BaseModel):
    """Mirrors the original ``model/config.py`` defaults."""

    min_faces_per_person: int = 30
    image_resize: float = 0.4
    test_size: float = 0.2
    random_state: int = 42
    n_components: int = 128
    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.3
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    promotion_f1_threshold: float = Field(
        default=0.0,
        description="Minimum F1 improvement over production to promote.",
    )
