#!/usr/bin/env python
"""Train the fruit quality model with either backend.

Usage:
    python scripts/train.py --backend tensorflow --config examples/config.yaml
    python scripts/train.py --backend pytorch    --config examples/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitqc.common.config import load_config, TrainConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fruit quality model")
    parser.add_argument("--backend", choices=["tensorflow", "pytorch"], required=True)
    parser.add_argument("--config", default=None, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else TrainConfig()

    if args.backend == "tensorflow":
        from fruitqc.tensorflow.train import train
    else:
        from fruitqc.pytorch.train import train

    train(cfg)


if __name__ == "__main__":
    main()
