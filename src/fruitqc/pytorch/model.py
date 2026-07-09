"""PyTorch model: shared backbone + three heads (fruit, age, quality).

Mirrors the TensorFlow architecture so the two backends are directly
comparable. ResNet is the default and best-performing backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common.config import TrainConfig


class _SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 3
        for out_ch in (32, 64, 128):
            layers += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 128

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x


class FruitQCModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        backbone = cfg.backbone.lower()

        if backbone == "resnet":
            from torchvision import models

            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            feat_dim = base.fc.in_features
            base.fc = nn.Identity()
            self.backbone = base
        elif backbone == "cnn":
            self.backbone = _SmallCNN()
            feat_dim = self.backbone.out_dim
        elif backbone == "rnn":
            self.backbone = _SmallCNN()  # CNN feature extractor…
            self.rnn = nn.GRU(self.backbone.out_dim, 128, batch_first=True)
            feat_dim = 128
        else:
            raise ValueError(f"Unknown backbone: {cfg.backbone!r}")

        self._is_rnn = backbone == "rnn"
        self.dropout = nn.Dropout(0.3)
        self.head_fruit = nn.Linear(feat_dim, cfg.num_fruit)
        self.head_age = nn.Linear(feat_dim, cfg.num_age)
        self.head_quality = nn.Linear(feat_dim, cfg.num_quality)

    def forward(self, x):
        feats = self.backbone(x)
        if self._is_rnn:
            feats = feats.unsqueeze(1)          # (B, 1, F) minimal sequence
            _, h = self.rnn(feats)
            feats = h.squeeze(0)
        feats = self.dropout(feats)
        return {
            "fruit": self.head_fruit(feats),
            "age": self.head_age(feats),
            "quality": self.head_quality(feats),
        }
