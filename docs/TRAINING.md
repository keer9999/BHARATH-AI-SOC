# YOLOv8 Training Guide

## Overview

This guide explains how to train, validate, and export the YOLOv8 waste classification model from scratch using the project's training pipeline.

---

## Table of Contents

1. [Dataset Preparation](#1-dataset-preparation)
2. [Training](#2-training)
3. [Evaluation & Validation](#3-evaluation--validation)
4. [Export for Edge Deployment](#4-export-for-edge-deployment)
5. [Custom Dataset Acquisition](#5-custom-dataset-acquisition)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)

---

## 1. Dataset Preparation

### Waste Class Taxonomy

| Class | Label | Bin Colour | Examples |
|-------|-------|------------|---------|
| `plastic` | 0 | Blue | Bottles, bags, packaging, containers |
| `glass` | 1 | Green | Bottles, jars, broken glass |
| `metal` | 2 | Yellow | Cans, foil, scrap metal |
| `paper` | 3 | Blue | Newspapers, cardboard, office paper |
| `organic` | 4 | Brown | Food waste, garden waste, compostable material |

### Dataset Sources

The project uses a custom dataset assembled from:
- **Roboflow Universe**: `waste-detection.v1i.yolov8` (primary)
- **Custom collected images**: device-captured waste images in Indian context
- **Augmented variants**: rotations, flips, brightness/contrast jitter

### Dataset Structure (YOLOv8 format)

```
waste-detection.v1i.yolov8/
├── data.yaml           ← Dataset config
├── train/
│   ├── images/         ← Training images (.jpg)
│   └── labels/         ← YOLO format annotations (.txt)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**`data.yaml`:**
```yaml
path: /path/to/waste-detection.v1i.yolov8
train: train/images
val: valid/images
test: test/images

nc: 5
names: ['plastic', 'glass', 'metal', 'paper', 'organic']
```

### Annotation Format

Each `.txt` label file contains one detection per line:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalized** (0.0–1.0) relative to image dimensions.

**Example** (plastic bottle at centre):
```
0 0.512 0.489 0.234 0.512
```

---

## 2. Training

### Quick Start

```bash
source .venv/bin/activate
cd "version one c2s/module"

python train_waste_model.py \
  --data "../../version one c2s/waste-detection.v1i.yolov8/data.yaml" \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device cpu
```

### Training Script Arguments

```python
# train_waste_model.py — key arguments
python train_waste_model.py \
  --data PATH        # Path to data.yaml
  --model NAME       # yolov8n.pt | yolov8s.pt | yolov8m.pt
  --epochs N         # Training epochs (default: 100)
  --imgsz N          # Input image size (default: 640)
  --batch N          # Batch size (default: 16; use 8 for RPi/low-RAM)
  --device cpu|0|1   # Compute device (cpu or GPU index)
  --workers N        # DataLoader workers (default: 4; use 2 on RPi)
  --project DIR      # Output directory (default: runs/detect)
  --name EXP         # Experiment name
  --resume           # Resume from last checkpoint
```

### Direct Ultralytics API

```python
from ultralytics import YOLO

# Load pretrained base model (auto-downloads from Ultralytics)
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data="waste-detection.v1i.yolov8/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu",          # Use 0 for CUDA GPU
    workers=4,
    project="runs/waste",
    name="yolov8n_v1",
    # Augmentation
    hsv_h=0.015,           # Hue jitter
    hsv_s=0.7,             # Saturation jitter
    hsv_v=0.4,             # Value jitter
    degrees=10.0,          # Rotation range
    translate=0.1,         # Translation fraction
    scale=0.5,             # Scale jitter
    flipud=0.0,            # Vertical flip probability
    fliplr=0.5,            # Horizontal flip probability
    mosaic=1.0,            # Mosaic augmentation
    mixup=0.1,             # MixUp augmentation
    # Training settings
    lr0=0.01,              # Initial learning rate
    lrf=0.01,              # Final learning rate factor
    warmup_epochs=3,       # Warmup epochs
    patience=50,           # Early stopping patience
)
```

### Training Output

Results are saved to `runs/waste/yolov8n_v1/`:
```
runs/waste/yolov8n_v1/
├── weights/
│   ├── best.pt        ← Best checkpoint (use this for inference)
│   └── last.pt        ← Latest checkpoint
├── results.csv        ← Per-epoch metrics
├── results.png        ← Training curves plot
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── val_batch*.jpg     ← Validation prediction samples
```

---

## 3. Evaluation & Validation

### Validate on Test Set

```python
from ultralytics import YOLO

model = YOLO("runs/waste/yolov8n_v1/weights/best.pt")
metrics = model.val(
    data="waste-detection.v1i.yolov8/data.yaml",
    split="test",
    imgsz=640,
    conf=0.5,
    iou=0.6,
)

print(f"mAP@0.5:     {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95:{metrics.box.map:.3f}")
print(f"Precision:   {metrics.box.mp:.3f}")
print(f"Recall:      {metrics.box.mr:.3f}")
```

### Expected Performance

| Model | mAP@0.5 | mAP@0.5:0.95 | RPi 5 FPS | Size |
|-------|---------|--------------|-----------|------|
| YOLOv8n | ~72% | ~52% | 18 FPS | 6.3 MB |
| YOLOv8s | ~78% | ~57% | 12 FPS | 22 MB |
| YOLOv8m | ~82% | ~62% | 6 FPS | 50 MB |

### Per-class Analysis

```python
# Print per-class AP
for i, cls in enumerate(['plastic', 'glass', 'metal', 'paper', 'organic']):
    ap50 = metrics.box.ap50[i]
    print(f"{cls:10s}: AP@0.5 = {ap50:.3f}")
```

### Benchmark Inference Speed on RPi 5

```python
import time
from ultralytics import YOLO
import cv2

model = YOLO("runs/waste/yolov8n_v1/weights/best.pt")
frame = cv2.imread("test_images/sample.jpg")

# Warmup
model.predict(frame, verbose=False)

# Benchmark
N = 50
start = time.perf_counter()
for _ in range(N):
    model.predict(frame, conf=0.5, verbose=False)
elapsed = time.perf_counter() - start
print(f"Avg inference: {elapsed/N*1000:.1f} ms ({N/elapsed:.1f} FPS)")
```

---

## 4. Export for Edge Deployment

### Export to ONNX (for Zynq-7000 / OpenCV DNN)

```python
from ultralytics import YOLO
model = YOLO("runs/waste/yolov8n_v1/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True, opset=12)
# → best.onnx
```

### Export to TFLite (for future mobile deployment)

```python
model.export(format="tflite", imgsz=640, int8=True)
# → best_int8.tflite
```

### Export to ExecuTorch (ARM-optimised)

```python
# Requires executorch package
model.export(format="executorch", imgsz=640)
# → best.pte
```

### Export to CoreML (macOS / iOS)

```python
model.export(format="coreml", imgsz=640, nms=True)
```

---

## 5. Custom Dataset Acquisition

### Roboflow (Recommended)

1. Go to [universe.roboflow.com](https://universe.roboflow.com)
2. Search for "waste detection" or "garbage classification"
3. Export in **YOLOv8** format
4. Download to `version one c2s/waste-detection.v1i.yolov8/`

### Collect Custom Images

```python
# Capture frames from camera for annotation
import cv2, os, time

cap = cv2.VideoCapture(0)
os.makedirs("data/collection", exist_ok=True)

count = 0
print("Press SPACE to capture, Q to quit")
while True:
    ret, frame = cap.read()
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)
    if key == ord(' '):
        path = f"data/collection/frame_{count:04d}.jpg"
        cv2.imwrite(path, frame)
        print(f"Saved: {path}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Annotate with LabelImg

```bash
pip install labelImg
labelImg data/collection/ data/classes.txt
```

---

## 6. Hyperparameter Tuning

### Auto-tune with Ray Tune

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
result = model.tune(
    data="waste-detection.v1i.yolov8/data.yaml",
    epochs=30,
    iterations=50,       # Number of tuning trials
    optimizer="AdamW",
    plots=True,
    save=True,
    val=True,
)
```

### Recommended Hyperparameters (Indian Waste Dataset)

```yaml
# Optimised for mixed indoor/outdoor Indian waste scenes
lr0: 0.008
lrf: 0.012
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
hsv_h: 0.025     # Higher hue jitter for varied lighting
hsv_s: 0.75
hsv_v: 0.5       # Higher value jitter for outdoor/indoor variation
degrees: 15.0    # More rotation for irregular bin placement
translate: 0.15
scale: 0.6
mosaic: 1.0
mixup: 0.1
copy_paste: 0.1  # Helps with class imbalance
```
