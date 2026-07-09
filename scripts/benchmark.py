#!/usr/bin/env python
"""Measure inference throughput (FPS) for the current device.

This produces the throughput half of the results table. Accuracy must be
measured separately on a labelled test split (see scripts/evaluate.py, to be
added). Reported figures in the README await verification against the paper.

Usage:
    python scripts/benchmark.py --backend pytorch --weights runs/best.pt --iters 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitqc.common.config import TrainConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark inference FPS")
    parser.add_argument("--backend", choices=["tensorflow", "pytorch"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    cfg = TrainConfig()
    dummy = np.random.rand(1, cfg.image_size, cfg.image_size, 3).astype("float32")

    if args.backend == "tensorflow":
        import tensorflow as tf

        model = tf.keras.models.load_model(args.weights)
        infer = lambda: model.predict(dummy, verbose=0)
    else:
        import torch
        from fruitqc.pytorch.model import FruitQCModel

        model = FruitQCModel(cfg)
        model.load_state_dict(torch.load(args.weights, map_location="cpu"))
        model.eval()
        t = torch.from_numpy(dummy.transpose(0, 3, 1, 2))
        infer = lambda: model(t)

    # warmup
    for _ in range(10):
        infer()

    start = time.time()
    for _ in range(args.iters):
        infer()
    elapsed = time.time() - start

    fps = args.iters / elapsed
    print(f"backend={args.backend}  iters={args.iters}  "
          f"elapsed={elapsed:.2f}s  throughput={fps:.1f} FPS")


if __name__ == "__main__":
    main()
