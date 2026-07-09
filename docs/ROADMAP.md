# Roadmap

Ordered roughly by priority.

## Near term
- [ ] **Re-run metrics on this reconstructed code** to reproduce the author-
      confirmed numbers and add a confusion matrix.
- [ ] Add real screenshots of the running demo to `images/` (placeholders now).
- [ ] Publish a small labelled sample set (or a download+prepare script) so the
      training commands run out of the box.

## Medium term
- [ ] Multi-object detection/localization (currently one fruit per frame).
- [ ] Quantization + TensorRT export to lift Raspberry Pi / Jetson throughput.
- [ ] Expand fruit classes beyond apple/banana/orange.
- [ ] Add `evaluate.py` outputs (per-class precision/recall) to CI artifacts.

## Long term
- [ ] On-device continual/active learning from the unlabelled stream.
- [ ] Compare additional lightweight backbones (MobileNetV3, EfficientNet-Lite).
- [ ] Package a ready-to-flash edge image.

Contributions welcome — see [CONTRIBUTING.md](../CONTRIBUTING.md).
