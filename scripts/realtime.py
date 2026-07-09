#!/usr/bin/env python
"""Run the real-time webcam demo.

Usage:
    python scripts/realtime.py --backend tensorflow --weights runs/best.h5 --camera 0
    python scripts/realtime.py --backend pytorch    --weights runs/best.pt --camera 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitqc.common.config import TrainConfig  # noqa: E402
from fruitqc.realtime.webcam import run  # noqa: E402


def _argmax_labels(cfg, fruit_logits, age_logits, quality_logits):
    return {
        "fruit": cfg.fruit_types[int(np.argmax(fruit_logits))],
        "age": cfg.age_stages[int(np.argmax(age_logits))],
        "quality": cfg.quality_classes[int(np.argmax(quality_logits))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime fruit quality demo")
    parser.add_argument("--backend", choices=["tensorflow", "pytorch"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cfg = TrainConfig()

    if args.backend == "tensorflow":
        import tensorflow as tf

        model = tf.keras.models.load_model(args.weights)

        def predict_fn(batch):
            f, a, q = model.predict(batch, verbose=0)
            return _argmax_labels(cfg, f[0], a[0], q[0])

    else:
        import torch
        from fruitqc.pytorch.model import FruitQCModel

        model = FruitQCModel(cfg)
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.eval()

        def predict_fn(batch):
            t = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float()
            with torch.no_grad():
                out = model(t)
            return _argmax_labels(
                cfg, out["fruit"][0].numpy(),
                out["age"][0].numpy(), out["quality"][0].numpy(),
            )

    run(cfg, predict_fn, camera=args.camera)


if __name__ == "__main__":
    main()
