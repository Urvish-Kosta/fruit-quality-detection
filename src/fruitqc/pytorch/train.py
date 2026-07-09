"""PyTorch Dataset + training loop."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from ..common.config import TrainConfig
from ..common.data import discover_samples
from ..common.preprocess import preprocess_frame
from .model import FruitQCModel


class FruitDataset(Dataset):
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.samples = discover_samples(cfg)
        if not self.samples:
            raise RuntimeError(
                f"No samples found under {cfg.data_dir!r}. See docs/DATASET.md."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2

        s = self.samples[idx]
        bgr = cv2.imread(s.path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        arr = preprocess_frame(rgb, self.cfg.image_size)          # HWC float
        tensor = torch.from_numpy(np.transpose(arr, (2, 0, 1)))   # CHW
        labels = {
            "fruit": torch.tensor(s.fruit_idx),
            "age": torch.tensor(s.age_idx),
            "quality": torch.tensor(s.quality_idx),
        }
        return tensor, labels


def train(cfg: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.output_dir, exist_ok=True)

    ds = FruitDataset(cfg)
    n_val = int(len(ds) * cfg.val_split)
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = FruitQCModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_dl:
            x = x.to(device)
            opt.zero_grad()
            out = model(x)
            loss = sum(criterion(out[k], y[k].to(device)) for k in out)
            loss.backward()
            opt.step()

        val_loss = _validate(model, val_dl, criterion, device)
        print(f"epoch {epoch + 1}/{cfg.epochs}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best.pt"))

    return model


@torch.no_grad()
def _validate(model, dl, criterion, device):
    model.eval()
    total = 0.0
    for x, y in dl:
        x = x.to(device)
        out = model(x)
        total += float(sum(criterion(out[k], y[k].to(device)) for k in out))
    return total / max(len(dl), 1)
