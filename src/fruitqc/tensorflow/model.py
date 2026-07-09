"""TensorFlow/Keras model: shared backbone + three output heads.

Backbones:
  - resnet : ResNet50 (ImageNet-pretrained) — best-performing, default
  - cnn    : small custom CNN — lightweight baseline
  - rnn    : CNN feature extractor feeding a recurrent layer — evaluated for
             comparison; included because the original study compared CNN, RNN,
             and ResNet.

The three heads (fruit type, age, quality) share the backbone so one forward
pass produces all predictions — important for edge throughput.
"""

from __future__ import annotations

from ..common.config import TrainConfig


def build_model(cfg: TrainConfig):
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(cfg.image_size, cfg.image_size, 3), name="image")

    backbone = cfg.backbone.lower()
    if backbone == "resnet":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
        )
        features = base.output
    elif backbone == "cnn":
        x = inp
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.MaxPooling2D()(x)
        features = layers.GlobalAveragePooling2D()(x)
    elif backbone == "rnn":
        x = inp
        for filters in (32, 64):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.MaxPooling2D()(x)
        # Treat rows as a sequence of feature vectors for the recurrent layer.
        shape = x.shape
        x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
        features = layers.GRU(128)(x)
    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone!r}")

    shared = layers.Dropout(0.3)(features)

    fruit_out = layers.Dense(cfg.num_fruit, activation="softmax", name="fruit")(shared)
    age_out = layers.Dense(cfg.num_age, activation="softmax", name="age")(shared)
    quality_out = layers.Dense(cfg.num_quality, activation="softmax", name="quality")(shared)

    return Model(inputs=inp, outputs=[fruit_out, age_out, quality_out], name="fruitqc")


def compile_model(model, cfg: TrainConfig):
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.learning_rate),
        loss={
            "fruit": "sparse_categorical_crossentropy",
            "age": "sparse_categorical_crossentropy",
            "quality": "sparse_categorical_crossentropy",
        },
        metrics={"fruit": "accuracy", "age": "accuracy", "quality": "accuracy"},
    )
    return model
