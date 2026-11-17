<div align="center">

<img src="https://img.shields.io/badge/Bharat%20AI--SoC-2026--27-FF6B00?style=for-the-badge&logo=arm&logoColor=white" alt="Bharat AI-SoC"/>

# EcoSort AI
### On-Device Multimodal Waste Segregation & Circular Recycling Platform

*Problem Statement 2 — AI-Powered Edge Waste Classification and Disposal Assistant*  
*Organised by Arm × MeitY C2S × IIT Delhi*

<br/>

[![CI](https://github.com/keer9999/BHARATH-AI-SOC/actions/workflows/ci.yml/badge.svg)](https://github.com/keer9999/BHARATH-AI-SOC/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8%20%7C%20YOLO11-00FFFF?logo=ultralytics&logoColor=black)](https://docs.ultralytics.com/)
[![Llama](https://img.shields.io/badge/LLM-Llama%203.2%201B-6C2FBF?logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![Platform](https://img.shields.io/badge/Edge-Zynq--7000%20%7C%20ARM%20Cortex--A9-0091BD?logo=xilinx&logoColor=white)](hardware/zynq7000/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/keer9999/BHARATH-AI-SOC?quickstart=1)

> **Click "Open in GitHub Codespaces" above to launch the full EcoSort AI web interface in one click — no local setup required.**

<br/>

</div>

---

## Overview

**EcoSort AI** is a fully on-device, multimodal AI system for real-time waste identification and smart disposal guidance. It runs the complete inference pipeline — vision *and* language — entirely on edge hardware (Zynq-7000 ARM Cortex-A9), eliminating cloud dependency and enabling deployment anywhere.

The system addresses **Problem Statement 2** of the **Bharat AI-SoC Challenge 2026-27**: build an AI-powered edge solution for automated waste classification and circular recycling advisory using ARM SoC platforms.

### What It Does

| Modality | Model | Output |
|---|---|---|
| 👁️ **Vision** | YOLOv8 / YOLO11 (66+ classes) | Detects waste type from live camera or uploaded image |
| 🤖 **Language** | Llama 3.2 1B / WasteAdvisor | Gives disposal instructions, recycling tips, hazard warnings |
| 💡 **Hardware** | Zynq-7000 ZedBoard LEDs (LD0–LD7) | Physically lights the correct bin for immediate sorting |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EcoSort AI Pipeline                          │
│                                                                     │
│  Camera / Image                                                     │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  Modality 1 │    │   BIN ROUTER     │    │  Modality 2      │   │
│  │  VISION     │───▶│  PLASTIC → BLUE  │───▶│  LANGUAGE        │   │
│  │             │    │  GLASS   → GREEN │    │                  │   │
│  │  YOLOv8     │    │  METAL   → GREY  │    │  Llama 3.2 1B   │   │
│  │  YOLO11s    │    │  PAPER   → WHITE │    │  WasteAdvisor    │   │
│  │  (66 class) │    │  ORGANIC → BROWN │    │  (2300 rules)    │   │
│  └─────────────┘    └──────────────────┘    └──────────────────┘   │
│                              │                       │              │
│                              ▼                       ▼              │
│                   ┌──────────────────┐   ┌──────────────────────┐  │
│                   │  Zynq-7000       │   │  Web Dashboard       │  │
│                   │  ZedBoard LEDs   │   │  Live Stream +       │  │
│                   │  LD1-LD7 → Bin  │   │  Analytics + Chat    │  │
│                   └──────────────────┘   └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Design Principle:** Both modalities run entirely on-device on the ARM Cortex-A9 cores of the Zynq-7000 SoC. No cloud calls, no network dependency, no latency from remote inference.

---

## The Four Detection Models

EcoSort AI uses an ensemble of four YOLO models for maximum coverage and accuracy:

| Model File | Architecture | Classes | Role |
|---|---|---|---|
| `best.pt` | YOLOv8s (22.5 MB) | 5 — Plastic, Glass, Metal, Paper/Cardboard, Organic | **Primary model** — powers ZedBoard LED bin routing |
| `ecosort_waste_yolo11s.pt` | YOLO11s (19.2 MB) | 66 — Aerosol, Battery, Tupperware, Drink Can… | **Fine-grained classifier** — feeds Llama disposal advisory |
| `weights.pt` | YOLOv8s (20.3 MB) | 1 — Bottle | **Bottle specialist** — boosts precision for containers |
| `yolov8n.pt` | YOLOv8n (6.5 MB) | 80 COCO + waste remap | **Fallback / dual-model** — via `COCO_WASTE_MAP` |

> **Ensemble Strategy:** `best.pt` handles primary bin routing. `ecosort_waste_yolo11s.pt` provides fine-grained classification for the language model's advisory. `weights.pt` runs in parallel for bottle-specific scenes. `yolov8n.pt` acts as a CPU-light fallback on constrained hardware.

---

## Zynq-7000 LED Bin Mapping

When running on the ZedBoard, on-board LEDs provide instant physical feedback for sorting:

| LED | Waste Type | Bin Colour | Examples |
|---|---|---|---|
| **LD1** | Plastic | 🔵 Blue | Bottles, bags, packaging |
| **LD2** | Glass | 🟢 Green | Jars, bottles, windows |
| **LD3** | Metal | ⚫ Grey | Cans, foil, wire |
| **LD4** | Paper / Cardboard | ⚪ White | Boxes, newspapers, cartons |
| **LD5** | Organic / Food | 🟤 Brown | Peels, food scraps, compost |
| **LD6** | E-Waste | 🔴 Red | Batteries, cables, PCBs |
| **LD7** | Hazardous | 🟠 Orange | Chemicals, aerosols, sharps |

---

## Quick Start

### Option A — GitHub Codespaces (Zero Setup, Recommended for Judges)

1. Click **[Open in GitHub Codespaces](https://codespaces.new/keer9999/BHARATH-AI-SOC?quickstart=1)**
2. Wait ~2 minutes for the environment to build and dependencies to install
3. The web UI opens automatically at port **5000**
4. Upload any waste image or use the AI chat to test the system

### Option B — Local Setup (Linux / macOS)

```bash
# 1. Clone the repository
git clone https://github.com/keer9999/BHARATH-AI-SOC.git
cd BHARATH-AI-SOC

# 2. Run the one-click installer (creates venv, installs all deps)
bash scripts/setup.sh

# 3. Download the YOLO model weights
bash scripts/download_models.sh

# 4. Launch the web interface
source .venv/bin/activate
python src/web_app.py --host 0.0.0.0 --port 5000
```

Open **http://localhost:5000** in your browser.

### Option C — CLI (Headless / Zynq-7000)

```bash
# Single image analysis — full multimodal report
python src/main.py --image /path/to/waste_image.jpg

# Live webcam stream with real-time segregation
python src/main.py --headless

# Self-test (no camera or image needed)
python src/main.py --test --headless

# Zynq-7000 on-board run with LED output
python hardware/zynq7000/bram/main_zynq.py --test
```

---

## Web Interface Features

The Flask web UI (`src/web_app.py`) provides:

| Feature | Description |
|---|---|
| 📷 **Live Webcam Stream** | MJPEG stream with real-time YOLO overlay and bin label |
| 🖼️ **Image Upload & Detect** | Drag-and-drop any waste photo → instant multimodal report |
| 💬 **AI Chat Assistant** | Ask disposal questions; powered by Llama 3.2 or WasteAdvisor |
| 📊 **Analytics Dashboard** | Session stats: items detected, bins sorted, confidence chart |
| 🌐 **REST API** | `/api/detect`, `/api/chat`, `/api/webcam/stream` endpoints |

---

## Project Structure

```
BHARATH-AI-SOC/
│
├── src/                        # Core application
│   ├── main.py                 # CLI entry point (image / webcam / test)
│   ├── web_app.py              # Flask web UI + REST API
│   ├── waste_detector.py       # YOLO ensemble inference engine
│   ├── waste_advisor.py        # WasteAdvisor (2,300 rule KB + Llama fallback)
│   ├── llama_engine.py         # Llama 3.2 GGUF loader (optional)
│   ├── config.py               # Model paths, thresholds, bin mapping
│   └── yolo/weights/           # YOLO .pt model files (gitignored, use download script)
│
├── hardware/
│   ├── zynq7000/
│   │   └── bram/
│   │       ├── main_zynq.py    # Zynq-7000 entry point with LED bin routing
│   │       └── config_zynq.py  # ZedBoard hardware configuration
│   └── hailo_npu/              # Hailo-8 NPU acceleration (optional)
│
├── models/                     # Model weights directory
│   ├── best.pt                 # Primary 5-class bin-sorting model
│   ├── ecosort_waste_yolo11s.pt# 66-class fine-grained model
│   ├── weights.pt              # Bottle specialist
│   └── yolov8n.pt              # COCO fallback model
│
├── scripts/
│   ├── setup.sh                # One-click environment installer
│   ├── download_models.sh      # Pull model weights from GitHub Releases
│   └── run_tests.sh            # Test suite runner
│
├── .devcontainer/
│   └── devcontainer.json       # GitHub Codespaces configuration
│
├── .github/workflows/
│   └── ci.yml                  # CI: Lint (Ruff) + Tests + Docker build
│
└── docs/                       # Extended documentation
```

---

## Technology Stack

| Component | Technology | Details |
|---|---|---|
| **Vision AI** | Ultralytics YOLOv8 / YOLO11 | Real-time object detection, 66-class fine-grained |
| **Language AI** | Meta Llama 3.2 1B (GGUF) | On-device disposal advisory, no cloud needed |
| **Advisory KB** | WasteAdvisor (custom) | 2,300 rules — resin codes, decomp times, DIY upcycling |
| **Web Framework** | Flask 3.x | REST API + MJPEG streaming + chatbot |
| **Edge Platform** | Zynq-7000 (ARM Cortex-A9) | Primary deployment target — ZedBoard |
| **Deep Learning** | PyTorch 2.x + ONNX | Model inference engine |
| **CI/CD** | GitHub Actions | Lint, test, Docker build on every push |
| **Dev Environment** | GitHub Codespaces | Zero-config browser-based launch |

---

## Deployment Targets

### Laptop / Desktop (Demo Mode)
Full web UI with webcam, image upload, analytics, and AI chat.

```bash
bash scripts/setup.sh && python src/web_app.py
```

### Zynq-7000 ZedBoard (Competition Target)
Bare-metal ARM inference with physical LED bin routing.

```bash
python hardware/zynq7000/bram/main_zynq.py
```

### GitHub Codespaces (Judge / Evaluator)
Zero-install browser demo — full web UI launches automatically.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/keer9999/BHARATH-AI-SOC?quickstart=1)

---

## Model Weights

The `.pt` model files are **not stored in this repository** (too large for Git). Download them with:

```bash
bash scripts/download_models.sh
```

Or place them manually in `src/yolo/weights/` and `models/`.

> Model weights for `best.pt` (the custom-trained 5-class primary model) are available as a **[GitHub Release asset](https://github.com/keer9999/BHARATH-AI-SOC/releases)**.

---

## Contributing

```bash
# Fork → branch → commit → PR
git checkout -b feature/your-feature
git commit -m "feat: description"
git push origin feature/your-feature
```

Please run `ruff check src tests` before submitting a PR.

---

## Competition Context

| Field | Details |
|---|---|
| **Challenge** | Bharat AI-SoC 2026-27 |
| **Problem Statement** | PS2 — AI-Powered Waste Classification & Disposal Assistant |
| **Organisers** | Arm × MeitY C2S Programme × IIT Delhi |
| **Target Platform** | Arm SoC (Zynq-7000, ARM Cortex-A9) |
| **Core Requirement** | On-device multimodal AI — Vision + Language, no cloud |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the Bharat AI-SoC Challenge 2026-27**  
*Arm × MeitY C2S × IIT Delhi*

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/keer9999/BHARATH-AI-SOC?quickstart=1)

</div>
