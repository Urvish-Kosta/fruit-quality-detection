#!/usr/bin/env python
"""Evaluate per-head accuracy (fruit, age, quality) on a held-out split.

Usage:
    python scripts/evaluate.py --backend pytorch --weights runs/best.pt --config examples/config.yaml
    python scripts/evaluate.py --backend tensorflow --weights runs/best.h5 --config examples/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitqc.common.config import load_config, TrainConfig  # noqa: E402
from fruitqc.common.data import discover_samples  # noqa: E402
from fruitqc.common.preprocess import preprocess_frame, to_batch  # noqa: E402


def _load_rgb(path):
    import cv2

    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-head accuracy")
    parser.add_argument("--backend", choices=["tensorflow", "pytorch"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else TrainConfig()
    samples = discover_samples(cfg)
    if not samples:
        raise RuntimeError(f"No samples under {cfg.data_dir!r}; see docs/DATASET.md")

    # Deterministic held-out tail matching the training val_split.
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(samples))
    n_val = int(len(samples) * cfg.val_split)
    val_idx = order[:n_val]

    if args.backend == "tensorflow":
        import tensorflow as tf

        model = tf.keras.models.load_model(args.weights)

        def predict(batch):
            f, a, q = model.predict(batch, verbose=0)
            return f[0], a[0], q[0]
    else:
        import torch
        from fruitqc.pytorch.model import FruitQCModel

        model = FruitQCModel(cfg)
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.eval()

        def predict(batch):
            t = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float()
            with torch.no_grad():
                out = model(t)
            return (out["fruit"][0].numpy(), out["age"][0].numpy(),
                    out["quality"][0].numpy())

    correct = {"fruit": 0, "age": 0, "quality": 0}
    for i in val_idx:
        s = samples[i]
        batch = to_batch(preprocess_frame(_load_rgb(s.path), cfg.image_size))
        f, a, q = predict(batch)
        correct["fruit"] += int(np.argmax(f) == s.fruit_idx)
        correct["age"] += int(np.argmax(a) == s.age_idx)
        correct["quality"] += int(np.argmax(q) == s.quality_idx)

    n = len(val_idx)
    print(f"Evaluated {n} samples")
    for head in ("fruit", "age", "quality"):
        print(f"  {head:8s} accuracy: {correct[head] / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
