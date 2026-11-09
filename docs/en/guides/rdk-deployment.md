# D-Robotics RDK X5 Deployment Guide

This guide explains how to export and run Ultralytics YOLO models on **D-Robotics RDK X5** hardware using the BPU (Brain Processing Unit).

## 1. Environment Setup (PC/Server)

The model conversion toolchain (`hb_mapper`) requires an x86_64 Linux environment.

```bash
# Install the D-Robotics Toolchain
pip install rdkx5-yolo-mapper -i https://mirrors.aliyun.com/pypi/simple/
```

## 2. Export to BPU format (`.bin`)

You can export your trained PyTorch model (`.pt`) to the D-Robotics BPU format (`.bin`) directly using the Ultralytics API.

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

# Export to RDK format
# This will apply BPU-specific optimizations and call hb_mapper
model.export(format="rdk", data="coco8.yaml")
```

The output will be a `*_rdk_model/` directory containing the compiled `.bin` and `metadata.yaml`.

## 3. Inference on RDK X5 Board

On the RDK X5 board, ensure you have `hbm_runtime` installed (usually pre-installed in the system image).

```python
from ultralytics import YOLO

# Load the exported BPU model directory or the .bin file inside it
model = YOLO("yolo26n_rdk_model")

# Run inference
results = model("image.jpg")

# Show results
results[0].show()
```

## 4. Key Features

- **Automated Toolchain**: One-click export from `.pt` to `.bin`.
- **BPU Optimization**: Automatic monkey-patching of model heads to NHWC layout for maximum hardware efficiency.
- **Logit-based Filtering**: High-performance post-processing that filters background anchors using raw logits, significantly reducing CPU overhead.
- **Unified Runtime**: Uses `hbm_runtime` for efficient model execution on the BPU.

---
For more details, visit [D-Robotics OpenExplore Documentation](https://developer.d-robotics.cc/).
