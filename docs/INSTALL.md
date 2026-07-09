# Installation

## Desktop / development machine

```bash
git clone https://github.com/Urvish-Kosta/fruit-quality-detection.git
cd fruit-quality-detection
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install only the backend you need. `requirements.txt` lists both TensorFlow and
PyTorch; comment out whichever you are not using.

## Raspberry Pi

1. Use 64-bit Raspberry Pi OS.
2. System packages for OpenCV:
   ```bash
   sudo apt update
   sudo apt install -y python3-opencv libatlas-base-dev
   ```
3. Create a venv and install the lighter dependency set (typically TensorFlow
   Lite or CPU PyTorch, plus OpenCV). Full desktop TensorFlow/PyTorch wheels may
   be heavy on the Pi — prefer the CPU/Lite builds.

## Jetson Nano

1. Flash the appropriate **JetPack** image (bundles CUDA/cuDNN/TensorRT).
2. Use the **NVIDIA-provided** TensorFlow or PyTorch wheels built for L4T —
   do **not** `pip install tensorflow` from PyPI on Jetson, as it will not use
   the GPU.
3. Install OpenCV (JetPack usually ships a build) and remaining pure-Python deps.

## Verify

```bash
python -c "import fruitqc, sys; print('fruitqc', fruitqc.__version__)"
```

Camera check:

```bash
ls /dev/video*        # camera should appear, e.g. /dev/video0
```

## Notes

- GPU acceleration on Jetson requires the vendor wheels above; PyPI wheels are
  CPU-only there.
- If OpenCV fails to open the camera, confirm permissions and the correct index
  (`--camera 0`, `1`, …).
