"""Dataset discovery and label parsing.

Expected on-disk layout (see docs/DATASET.md). Each image lives under a folder
encoding its three labels:

    data/
      apple/ripe/good/img001.jpg
      apple/overripe/defective/img002.jpg
      banana/unripe/good/img003.jpg
      ...

i.e.  data/<fruit>/<age>/<quality>/<file>.jpg

This is a convention, not a hard requirement — swap in your own loader if your
Kaggle export is organized differently.
"""

from __future__ import annotations

import os
from typing import List, NamedTuple

from .config import TrainConfig


class Sample(NamedTuple):
    path: str
    fruit_idx: int
    age_idx: int
    quality_idx: int


def discover_samples(cfg: TrainConfig) -> List[Sample]:
    """Walk the dataset directory and build a list of labelled samples."""
    fruit_map = {n: i for i, n in enumerate(cfg.fruit_types)}
    age_map = {n: i for i, n in enumerate(cfg.age_stages)}
    quality_map = {n: i for i, n in enumerate(cfg.quality_classes)}

    samples: List[Sample] = []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for root, _dirs, files in os.walk(cfg.data_dir):
        parts = os.path.relpath(root, cfg.data_dir).split(os.sep)
        if len(parts) != 3:
            continue
        fruit, age, quality = parts
        if fruit not in fruit_map or age not in age_map or quality not in quality_map:
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                samples.append(
                    Sample(
                        path=os.path.join(root, f),
                        fruit_idx=fruit_map[fruit],
                        age_idx=age_map[age],
                        quality_idx=quality_map[quality],
                    )
                )
    return samples
