# EcoSort AI: Source Architecture

This directory houses the core application source code for EcoSort AI, engineered for the Bharat AI-SoC Challenge 2026-27 (Problem Statement 2).

The architecture provides fully on-device multimodal waste segregation:
1. Perception Modality (Vision): Real-time YOLOv8s / YOLO11s detection across 66 fine-grained waste classes and 5 primary bin categories, augmented by Slicing Aided Hyper Inference (SAHI) with Weighted Box Fusion (WBF).
2. Reasoning Modality (Language): On-device Meta Llama 3.2 1B (with embedded WasteAdvisor 2,300-rule knowledge engine) generating circular economy guidance, resin code identification, decomposition timelines, and upcycling solutions.
3. Hardware Actuation: Low-latency bin classification mapped directly to ZedBoard Zynq-7000 AXI GPIO LEDs (LD1-LD7) and web UI telemetry indicators.

---

## Directory Layout

```
src/
├── main.py                     # Primary CLI entry point (camera, single-image, self-test)
├── web_app.py                  # Flask web dashboard and REST API backend
├── waste_detector.py           # Multi-model YOLO inference engine and HUD overlay
├── sahi_detector.py            # Slicing Aided Hyper Inference (SAHI) and WBF engine
├── waste_advisor.py            # 2,300-rule recycling and circular economy knowledge engine
├── llama_engine.py             # On-device Llama 3.2 GGUF inference interface
├── executorch_llama_engine.py  # ExecuTorch mobile and embedded runtime loader
├── config.py                   # System thresholds, bin mappings, and model paths
├── database.py                 # SQLite persistence for detection events and telemetry
├── report_generator.py         # Automated PDF, CSV, and JSON audit generator
├── webcam_capture.py           # Threaded OpenCV camera acquisition hub
├── train_waste_model.py        # YOLO fine-tuning and transfer learning utility
├── web/                        # Web dashboard frontend
│   ├── templates/
│   │   ├── index.html          # Responsive web interface with SAHI control panel
│   │   └── 404.html            # Error view
│   └── static/
│       ├── css/style.css       # Interface styling and SAHI badge styles
│       └── js/
│           ├── app.js          # Core frontend interaction logic
│           └── sahi.js         # SAHI controller, status polling, and toggle logic
└── yolo/
    └── weights/                # Local directory for specialist YOLO model weights (.pt)
```

---

## Execution Modes

### 1. Single Image Analysis (CLI)

```bash
python src/main.py --image path/to/image.jpg
```

Executes YOLO detection, outputs class confidence, maps to corresponding disposal category, and generates full disposal advice.

### 2. Live Camera Mode (CLI)

```bash
python src/main.py --headless
```

Executes continuous detection on `/dev/video0`, streaming classification and bin routing decisions to console output.

### 3. Web Dashboard (Interactive)

```bash
# Standard single-pass execution
python src/web_app.py --host 0.0.0.0 --port 5000

# With SAHI sliced hyper-inference enabled
python src/web_app.py --host 0.0.0.0 --port 5000 --sahi --sahi-profile default
```

Launches the web UI with:
- Live camera stream with real-time bounding boxes
- Drag-and-drop image upload
- Interactive SAHI toggle and hardware profile switcher
- Multimodal AI assistant for disposal and upcycling inquiries
- Historical analytics and category segregation statistics

### 4. Self-Test and Regression Suite

```bash
python src/main.py --self-test
```

Executes an end-to-end regression across vision, language, and persistence subsystems without camera or display dependencies.
