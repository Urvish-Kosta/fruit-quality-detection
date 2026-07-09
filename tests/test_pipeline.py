"""Unit tests for the framework-agnostic pipeline.

These deliberately avoid importing TensorFlow/PyTorch so they run fast in CI
without the heavy deps.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fruitqc.common import (  # noqa: E402
    TrainConfig,
    preprocess_frame,
    to_batch,
    discover_samples,
)


def test_config_label_counts():
    cfg = TrainConfig()
    assert cfg.num_fruit == 3
    assert cfg.num_age == 3
    assert cfg.num_quality == 2


def test_preprocess_shape_and_range():
    img = (np.random.rand(300, 400, 3) * 255).astype("uint8")
    out = preprocess_frame(img, 224, normalize=False)
    assert out.shape == (224, 224, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_to_batch_adds_dim():
    img = np.zeros((224, 224, 3), dtype="float32")
    assert to_batch(img).shape == (1, 224, 224, 3)


def test_discover_samples(tmp_path):
    import cv2

    p = tmp_path / "banana" / "overripe" / "defective"
    p.mkdir(parents=True)
    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    cv2.imwrite(str(p / "a.jpg"), img)

    cfg = TrainConfig(data_dir=str(tmp_path))
    samples = discover_samples(cfg)
    assert len(samples) == 1
    s = samples[0]
    assert s.fruit_idx == 1      # banana
    assert s.age_idx == 2        # overripe
    assert s.quality_idx == 1    # defective
