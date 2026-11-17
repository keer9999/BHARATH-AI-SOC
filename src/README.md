# EcoSort AI — Source (`src/`) Architecture

This directory contains the core application source code for **EcoSort AI**, developed for the **Bharat AI-SoC Challenge 2026-27 (Problem Statement 2)**.

The system performs **pure on-device multimodal waste segregation**:
1. **Modality 1 (Vision):** Real-time YOLOv8 / YOLO11 detection across 66 fine-grained waste classes and 5 primary bin categories.
2. **Modality 2 (Language):** On-device Llama 3.2 1B (with offline `WasteAdvisor` 2,300-rule knowledge engine) providing recycling guidelines, hazard warnings, decomposition timelines, and upcycling ideas.
3. **Hardware Routing:** Direct bin classification mapped to ZedBoard Zynq-7000 on-board LEDs (LD1–LD7) and web UI indicators.

---

## Directory Layout

```
src/
├── main.py                 # Primary CLI entry point (camera, single-image, self-test)
├── web_app.py              # Flask Web UI & REST API backend (live stream, chat, upload)
├── waste_detector.py       # Multi-model YOLO inference engine & HUD overlay
├── waste_advisor.py        # 2,300-rule recycling & circular economy knowledge engine
├── llama_engine.py         # On-device Llama 3.2 GGUF inference interface
├── executorch_llama_engine.py # ExecuTorch mobile/embedded runtime loader
├── config.py               # System thresholds, bin mappings, model paths
├── database.py             # SQLite persistence for detection events & analytics
├── report_generator.py     # PDF & JSON waste segregation audit generator
├── webcam_capture.py       # Threaded OpenCV camera capture
├── train_waste_model.py    # YOLO fine-tuning & transfer learning utility
├── web/                    # Web dashboard frontend
│   ├── templates/          # HTML templates (index.html, 404.html)
│   └── static/             # CSS, JavaScript (app.js), icons
└── yolo/
    └── weights/            # Local directory for YOLO .pt model weights
```

---

## Execution Modes

### 1. Single Image Analysis (CLI)
```bash
python src/main.py --image path/to/image.jpg
```
Runs YOLO detection, prints class confidence, maps to colored bin, and outputs full Llama disposal advisory.

### 2. Live Camera Mode (CLI)
```bash
python src/main.py --headless
```
Runs continuous detection on `/dev/video0`, streaming classification and bin routing to console logs.

### 3. Web Dashboard (Interactive)
```bash
python src/web_app.py --host 0.0.0.0 --port 5000
```
Launches web UI at `http://localhost:5000` with:
- Live camera stream with AI bounding boxes
- Drag-and-drop image upload
- Multimodal AI chatbot for disposal queries
- Historical analytics & bin segregation counts

### 4. Self-Test Mode
```bash
python src/main.py --test --headless
```
Verifies model loading, inference pipeline, and advisory generation without requiring a physical camera.

---

## Bin Routing Reference

| Waste Class | Target Bin | Color Code |
|---|---|---|
| Plastic | Blue | `#1E88E5` |
| Glass | Green | `#43A047` |
| Metal | Grey | `#757575` |
| Paper / Cardboard | White | `#F5F5F5` |
| Organic / Compost | Brown | `#6D4C41` |
| E-Waste | Red | `#E53935` |
| Hazardous | Orange | `#FB8C00` |
