"""Framework-agnostic building blocks: config, preprocessing, data discovery."""

from .config import TrainConfig, load_config, FRUIT_TYPES, AGE_STAGES, QUALITY_CLASSES
from .preprocess import preprocess_frame, to_batch
from .data import Sample, discover_samples

__all__ = [
    "TrainConfig",
    "load_config",
    "FRUIT_TYPES",
    "AGE_STAGES",
    "QUALITY_CLASSES",
    "preprocess_frame",
    "to_batch",
    "Sample",
    "discover_samples",
]
