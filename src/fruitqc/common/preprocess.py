"""Framework-agnostic image preprocessing.

Kept deliberately small: a single resize+normalize path used identically at
train time and at inference time on the edge device, so there is no train/serve
skew.
"""

from __future__ import annotations

import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_frame(image: np.ndarray, size: int, normalize: bool = True) -> np.ndarray:
    """Resize an HxWx3 uint8 image to (size, size, 3) float32 in [0, 1].

    Args:
        image: RGB image array (uint8).
        size: target square edge length.
        normalize: if True, apply ImageNet mean/std standardization
            (recommended for ResNet backbones).

    Returns:
        float32 array shaped (size, size, 3).
    """
    import cv2  # imported lazily so non-vision unit tests stay light

    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    if normalize:
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def to_batch(image: np.ndarray) -> np.ndarray:
    """Add a leading batch dimension: (H, W, C) -> (1, H, W, C)."""
    return np.expand_dims(image, axis=0)
