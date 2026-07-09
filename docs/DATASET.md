# Dataset

## Source

Training data was drawn from public **Kaggle** fruit image datasets covering
**apple, banana, and orange**. Both pre-labelled ("classified") and unlabelled
("unclassified") subsets were used: the model was first trained on the labelled
data, then retrained incorporating the additional data to improve robustness.

> Datasets are **not** redistributed in this repository. Download them from
> Kaggle under their respective licenses and prepare them locally.

## Expected layout

The loader (`fruitqc.common.data.discover_samples`) expects a three-level folder
convention encoding all three labels:

```
data/
├── apple/
│   ├── unripe/
│   │   ├── good/        *.jpg
│   │   └── defective/   *.jpg
│   ├── ripe/
│   │   ├── good/
│   │   └── defective/
│   └── overripe/
│       ├── good/
│       └── defective/
├── banana/
│   └── ... (same structure)
└── orange/
    └── ... (same structure)
```

Path pattern: `data/<fruit>/<age>/<quality>/<file>`

- `<fruit>` ∈ {apple, banana, orange}
- `<age>`   ∈ {unripe, ripe, overripe}
- `<quality>` ∈ {good, defective}

These label sets are defined in
[`src/fruitqc/common/config.py`](../src/fruitqc/common/config.py) — edit there if
your classes differ, and the change propagates to both backends.

## Preparing Kaggle data

Kaggle fruit datasets vary in how they label ripeness/quality. You will likely
need a short preparation step to sort raw images into the folders above. If the
raw set only distinguishes "fresh" vs "rotten", map those onto the `quality`
axis and assign `age` as best fits the source labels.

> If your prepared classes differ from the defaults, update `config.py` and
> [DATASET.md](DATASET.md) together so they stay consistent.

## Notes on class balance

- Keep the three classes reasonably balanced per fruit where possible.
- The unlabelled subset was used to expand coverage; document any manual
  labelling you do so results remain reproducible.
