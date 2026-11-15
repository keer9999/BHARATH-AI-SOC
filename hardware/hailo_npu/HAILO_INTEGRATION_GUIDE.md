# Hailo-8 / Hailo-8L Acceleration Re-Integration Guide

This archive contains all the code, tests, and tools built for **Hailo AI hardware acceleration (Hailo-8 / Hailo-8L)**.
Once your model training is completed, you can easily restore and deploy this acceleration pipeline.

---

## 📁 Archived Components

| File | Purpose |
| :--- | :--- |
| `inference_backend.py` | Dual-mode modular inference engine (`DEV` PyTorch, `ONNX` static runtime, `HAILO` hardware virtual streams, `MOCK`) |
| `hailo_export.py` | End-to-end export pipeline: PyTorch checkpoint -> fixed-shape ONNX -> Hailo Dataflow Compiler (DFC) INT8 quantization -> `.hef` binary |
| `test_hailo_backend.py` | Pytest suite validating letterbox decoding, NMS, mock tensors, ONNX session, and Hailo error handling |
| `weights.onnx` | Reference fixed-shape 640x640 ONNX model previously exported |

---

## 🚀 Step-by-Step Instructions After Model Training Finishes

### Step 1: Copy Files Back into the Active Module
Once your training has produced a final checkpoint (e.g., `yolo/runs/waste_detection/train/weights/best.pt` or `module/yolo/weights/weights.pt`), copy the backend files back:

```bash
# From P-yesist-v2/P-yesist-v2:
cp hailo_acceleration_archive/inference_backend.py module/
cp hailo_acceleration_archive/hailo_export.py module/
cp hailo_acceleration_archive/test_hailo_backend.py module/tests/
```

### Step 2: Export Your Trained PyTorch Weights to Fixed-Shape ONNX
The Hailo Dataflow Compiler requires a static-shape ONNX model (640x640):

```bash
cd module
python hailo_export.py --step onnx --weights yolo/weights/weights.pt --output yolo/weights/weights.onnx
```

### Step 3: Quantize & Compile to Hailo HEF Binary
On your Linux x86_64 machine or Docker container with the Hailo Dataflow Compiler (`hailo_sdk_client` installed):

```bash
python hailo_export.py --step hef --arch hailo8 --onnx yolo/weights/weights.onnx --output yolo/weights/weights.hef --calib-dir ../test_images/
```

*Note: For Raspberry Pi AI Kit (Hailo-8L), set `--arch hailo8l`.*

### Step 4: Run Hardware Accelerated Inference
On your edge deployment target (e.g. Raspberry Pi 5 with Hailo-8 M.2 HAT or Jetson):

```bash
# Run with Hailo AI NPU acceleration
python main.py --backend hailo --hef yolo/weights/weights.hef

# Or set via environment variable:
export WASTE_INFERENCE_BACKEND=hailo
python main.py
```

---

## 💡 Supported Inference Modes
- `DEV` (Default on PC): Runs PyTorch GPU/CPU inference directly with `ultralytics`.
- `ONNX`: Uses `onnxruntime` to test the exported static graph on PC before compiling to HEF.
- `HAILO`: Uses `hailo_platform` (HailoRT VDevice / virtual streams) with hardware INT8 acceleration.
- `MOCK`: Hardware-independent dry-run mode for CI/CD testing.
