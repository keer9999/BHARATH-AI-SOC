# Models Directory

This directory stores on-device AI model weights and artifacts for the Bharat AI-SoC Waste Management system.

## Included Models (Out-of-the-Box)

| File | Size | Task / Description | Format | Target |
| :--- | :--- | :--- | :--- | :--- |
| `best.pt` | ~22.5 MB | Custom-trained 5-class/7-class Waste Detection model | PyTorch / YOLO | Host PC / Edge CPU / SoC |
| `yolov8n.pt` | ~6.5 MB | General YOLOv8 Nano multi-class base model | PyTorch / YOLO | Host PC / Edge CPU / SoC |

## Llama 3.2 1B Instruct (Mandatory Challenge Requirement)

Due to file size limits (>100 MB), the quantized Llama 3.2 GGUF weights are downloaded on demand rather than stored in git.

### Automatic Download
Run the project model downloader script from the repository root:
```bash
bash scripts/download_models.sh
```
Or use the Python downloader:
```bash
python3 src/download_model.py
```

### Model Specifications
- **Model:** Llama 3.2 1B Instruct (`llama-3.2-1b-instruct-q4_k_m.gguf`)
- **Quantization:** Q4_K_M (4-bit medium quantization)
- **File Size:** ~807 MB
- **Target Memory Footprint:** < 1.2 GB RAM (ideal for Cortex-A9 / Cortex-A53 / Cortex-A72 Arm SoCs)
- **Hugging Face Source:** `bartowski/Llama-3.2-1B-Instruct-GGUF`

## Directory Hierarchy & Sync

When deploying or developing across different targets:
- `models/`: Root staging directory for all models
- `src/models/`: Runtime directory used by Llama engines (`llama_engine.py`)
- `src/yolo/weights/`: Runtime directory used by YOLO detectors (`waste_detector.py`)
- `hardware/zynq7000/external/models/`: Deployed to `/mnt/ext_storage/models/` on ZedBoard
- `hardware/zc702/external/models/`: Deployed to `/mnt/models/` on ZC702
