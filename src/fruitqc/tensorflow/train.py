"""TensorFlow data pipeline + training entry point."""

from __future__ import annotations

import os

from ..common.config import TrainConfig
from ..common.data import discover_samples
from .model import build_model, compile_model


def _build_dataset(cfg: TrainConfig):
    import tensorflow as tf

    samples = discover_samples(cfg)
    if not samples:
        raise RuntimeError(
            f"No samples found under {cfg.data_dir!r}. "
            "See docs/DATASET.md for the expected layout."
        )

    paths = [s.path for s in samples]
    fruit = [s.fruit_idx for s in samples]
    age = [s.age_idx for s in samples]
    quality = [s.quality_idx for s in samples]

    def _load(path, f, a, q):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, (cfg.image_size, cfg.image_size))
        img = tf.cast(img, tf.float32) / 255.0
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return img, {"fruit": f, "age": a, "quality": q}

    ds = tf.data.Dataset.from_tensor_slices((paths, fruit, age, quality))
    ds = ds.shuffle(len(paths), seed=cfg.seed)

    n_val = int(len(paths) * cfg.val_split)
    val_ds = ds.take(n_val).map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = ds.skip(n_val).map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def train(cfg: TrainConfig):
    import tensorflow as tf

    os.makedirs(cfg.output_dir, exist_ok=True)
    train_ds, val_ds = _build_dataset(cfg)

    model = compile_model(build_model(cfg), cfg)
    ckpt = os.path.join(cfg.output_dir, "best.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=cfg.epochs, callbacks=callbacks
    )
    model.save(os.path.join(cfg.output_dir, "final.h5"))
    return history
