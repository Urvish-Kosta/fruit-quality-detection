"""Central configuration: label spaces, image size, and training defaults.

Editing the label lists here propagates to both backends and the realtime demo,
so the three outputs (fruit type, age, quality) stay consistent everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# --- Label spaces -----------------------------------------------------------
# The deployed model produces three outputs from one forward pass.
# Adjust these to match the exact classes present in your prepared dataset.

FRUIT_TYPES: List[str] = ["apple", "banana", "orange"]
AGE_STAGES: List[str] = ["unripe", "ripe", "overripe"]
QUALITY_CLASSES: List[str] = ["good", "defective"]


@dataclass
class TrainConfig:
    """Training hyperparameters. Loaded from YAML in examples/config.yaml."""

    data_dir: str = "data"
    image_size: int = 224           # ResNet-friendly default
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-4
    val_split: float = 0.15
    seed: int = 42
    backbone: str = "resnet"        # one of: cnn | rnn | resnet
    output_dir: str = "runs"

    fruit_types: List[str] = field(default_factory=lambda: list(FRUIT_TYPES))
    age_stages: List[str] = field(default_factory=lambda: list(AGE_STAGES))
    quality_classes: List[str] = field(default_factory=lambda: list(QUALITY_CLASSES))

    @property
    def num_fruit(self) -> int:
        return len(self.fruit_types)

    @property
    def num_age(self) -> int:
        return len(self.age_stages)

    @property
    def num_quality(self) -> int:
        return len(self.quality_classes)


def load_config(path: str) -> TrainConfig:
    """Load a TrainConfig from a YAML file, falling back to defaults."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return TrainConfig(**raw)
