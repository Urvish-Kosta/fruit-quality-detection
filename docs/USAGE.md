# Usage

All commands assume the virtual environment is active and you are in the repo
root.

## 1. Prepare data

Organize images into the folder convention (see [DATASET.md](DATASET.md)):

```
data/<fruit>/<age>/<quality>/<image>.jpg
```

## 2. Configure

Edit [`examples/config.yaml`](../examples/config.yaml) to point at your `data/`
directory and choose a backbone (`resnet`, `cnn`, or `rnn`).

## 3. Train

TensorFlow:
```bash
python scripts/train.py --backend tensorflow --config examples/config.yaml
```

PyTorch:
```bash
python scripts/train.py --backend pytorch --config examples/config.yaml
```

Checkpoints are written to the configured `output_dir` (default `runs/`):
`best.h5` / `final.h5` (TF) or `best.pt` (PyTorch).

## 4. Evaluate accuracy

```bash
python scripts/evaluate.py --backend pytorch --weights runs/best.pt --config examples/config.yaml
```

Prints per-head accuracy (fruit, age, quality) on a held-out split.

## 5. Benchmark throughput (FPS)

Run **on the target device** to get its real FPS:

```bash
python scripts/benchmark.py --backend pytorch --weights runs/best.pt --iters 200
```

## 6. Real-time webcam demo

```bash
python scripts/realtime.py --backend tensorflow --weights runs/best.h5 --camera 0
```

An annotated window shows the predicted fruit, age, quality, and live FPS.
Press **q** to quit.

## Tips

- Plain, evenly-lit backgrounds improve accuracy noticeably.
- One fruit per frame — the model does not localize multiple objects.
- If FPS is low on the Pi, try the `cnn` backbone or a smaller `image_size`.
