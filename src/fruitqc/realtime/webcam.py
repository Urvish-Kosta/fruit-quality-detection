"""Real-time webcam inference with an annotated overlay.

Backend-agnostic: pass a `predict_fn` that maps a preprocessed batch to a dict
of {"fruit", "age", "quality"} label strings. The two backends provide thin
adapters (see scripts/realtime.py).
"""

from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np

from ..common.config import TrainConfig
from ..common.preprocess import preprocess_frame, to_batch


PredictFn = Callable[[np.ndarray], Dict[str, str]]


def run(cfg: TrainConfig, predict_fn: PredictFn, camera: int = 0) -> None:
    import cv2

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera}.")

    fps = 0.0
    prev = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch = to_batch(preprocess_frame(rgb, cfg.image_size))
            preds = predict_fn(batch)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            label = (
                f"{preds['fruit']} | age: {preds['age']} | "
                f"quality: {preds['quality']}  ({fps:4.1f} FPS)"
            )
            cv2.putText(
                frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.imshow("Fruit Quality Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
