<div align="center">

# Bharat AI-SoC Challenge 2026-27 | Problem Statement 2
### AI-Powered Edge Waste Classification and Disposal Assistant

[![CI](https://github.com/keer9999/bharath-ai-soc/actions/workflows/ci.yml/badge.svg)](https://github.com/keer9999/bharath-ai-soc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20%7C%20Zynq--7000-red?logo=raspberrypi&logoColor=white)](docs/HARDWARE_SETUP.md)
[![Vision](https://img.shields.io/badge/Vision-YOLOv8-00FFFF?logo=ultralytics)](https://docs.ultralytics.com/)
[![LLM](https://img.shields.io/badge/LLM-Llama%203.2%201B-6C2FBF)](https://ai.meta.com/llama/)
[![Architecture](https://img.shields.io/badge/Arch-Arm%20%7C%20FPGA-0091BD?logo=arm&logoColor=white)](https://www.arm.com/)
[![MeitY](https://img.shields.io/badge/Organised%20by-MeitY%20C2S-FF6B00)](https://www.meity.gov.in/)

<br/>

> **High-Performance Edge-AI Waste Management System** designed for embedded Arm and SoC architectures. Integrates real-time YOLOv8 object detection, local on-device Llama 3.2 1B advisory intelligence, multi-sensor environmental telemetry, and FPGA coprocessing for automated sorting and disposal guidance.

<br/>

**[Documentation](docs/)** | **[Architecture](docs/ARCHITECTURE.md)** | **[Hardware Setup](docs/HARDWARE_SETUP.md)** | **[Deployment](docs/DEPLOYMENT.md)** | **[Contributing](CONTRIBUTING.md)**

</div>

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Capabilities](#key-capabilities)
- [Technology Stack](#technology-stack)
- [Quick Start](#quick-start)
- [Hardware Setup](#hardware-setup)
- [API Reference](#api-reference)
- [Model Training and Optimization](#model-training-and-optimization)
- [Repository Structure](#repository-structure)
- [Deployment Options](#deployment-options)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Bharat AI-SoC Waste Management System** is an industrial-grade edge solution developed for the **Bharat AI-SoC Challenge 2026–27 (Problem Statement 2: AI-Powered Waste Classification & Disposal Assistant)**, organised by **Arm, MeitY C2S Programme, and IIT Delhi**.

The system addresses the challenge of municipal and industrial waste segregation by executing real-time computer vision inference and generative language reasoning entirely on-device without cloud dependencies.

### Primary Objectives
- **Material Detection & Segregation**: Rapid edge classification across primary categories (Plastics, Glass, Metals, Paper/Cardboard, Organic).
- **On-Device Advisory Intelligence**: Domain-specialized Llama 3.2 1B engine providing recycling guidelines, hazard warnings, and decomposition facts in English and Hindi.
- **Multi-Sensor Telemetry Fusion**: Environmental sensing array (fill level, gas concentration, humidity, moisture) interfaced via Raspberry Pi Pico W.
- **Heterogeneous Hardware Acceleration**: Optimized execution across Arm Cortex-A CPU, Hailo-8 NPU, and Xilinx Zynq-7000 FPGA platforms.

---

## System Architecture

```
+-------------------------------------------------------------------------+
|                  BHARAT AI-SoC EDGE INTELLIGENCE SYSTEM                 |
+-------------------------------------------------------------------------+

  +---------------------------------------------------------------------+
  |                           SENSING LAYER                             |
  |  Camera (USB/CSI)  HC-SR04 (Distance)  DHT11 (Temp/Hum)  MQ-135(Gas)|
  +-----------------------------------+---------------------------------+
                                      | Serial UART / USB
                                      v
  +---------------------------------------------------------------------+
  |                      EDGE PROCESSING PLATFORM                       |
  |               (Raspberry Pi 5 / Xilinx Zynq-7000 SoC)               |
  |                                                                     |
  |  +------------------------+      +-------------------------------+  |
  |  |    VISION PIPELINE     |      |       ADVISORY PIPELINE       |  |
  |  |  YOLOv8 Edge Engine    |      |   Llama 3.2 1B (GGUF / PTE)   |  |
  |  |  Dual-model priority   |      |   WasteAdvisor Knowledge Base |  |
  |  +-----------+------------+      +---------------+---------------+  |
  |              |                                   |                  |
  |              v                                   v                  |
  |  +---------------------------------------------------------------+  |
  |  |                  MULTI-SENSOR DATA FUSION                     |  |
  |  |    Material Class + Moisture + VOC/Gas + Fill Level           |  |
  |  +-------------------------------+-------------------------------+  |
  |                                  |                                  |
  |                                  v                                  |
  |  +---------------------------------------------------------------+  |
  |  |                  ON-DEVICE DATA & SAFETY                      |  |
  |  |      SQLite Telemetry Store  |  Safety Alert Engine (SMS/Mail)|  |
  |  +---------------------------------------------------------------+  |
  +-----------------------------------+---------------------------------+
                                      |
                   +------------------+------------------+
                   v                                     v
  +--------------------------------+   +--------------------------------+
  |        WEB APPLICATION         |   |      DESKTOP HUD CONSOLE       |
  |   Flask REST API + SSE Stream  |   |    OpenCV Real-time Overlay    |
  |   Live MJPEG Camera Feed       |   |    Low-latency Edge Display    |
  |   Llama 3.2 Interactive Chat   |   |    Autonomous Local Operation  |
  +--------------------------------+   +--------------------------------+
```

---

## Key Capabilities

<table>
<tr>
<td width="50%">

### Vision Intelligence
- **Real-Time Detection**: >=15 FPS on Arm Cortex-A76 (Raspberry Pi 5).
- **Multi-Class Taxonomy**: Plastic, Glass, Metal, Cardboard/Paper, Organic.
- **Dual-Model Inference**: Specialized waste detector cross-referenced with YOLOv8 backbone.
- **Configurable Confidence Gates**: Adaptable thresholding to reduce false positives.

</td>
<td width="50%">

### Sensor Telemetry
- **Fill-Level Monitoring**: Ultrasonic sensor telemetry with capacity alerts.
- **Hazard Identification**: Gas sensor (MQ series) for fermentation / hazardous vapors.
- **Moisture Gating**: Differentiates wet organic matter from dry paper/plastic.
- **Firmware**: MicroPython on Raspberry Pi Pico W via robust JSON serial protocol.

</td>
</tr>
<tr>
<td width="50%">

### Advisory & Reasoning
- **On-Device LLM**: Llama 3.2 1B Instruct quantized via `llama-cpp-python` (GGUF).
- **ExecuTorch Integration**: Reference runtime targeting Arm SoC acceleration.
- **Expert System**: Rule-based knowledge base spanning 2,300+ lines for sub-5ms guidance.
- **Bilingual Interface**: Native support for English and Hindi queries.

</td>
<td width="50%">

### Operations & Telemetry
- **Full-Stack Web Console**: Real-time MJPEG camera stream and telemetry dashboard.
- **Automated Reporting**: Export comprehensive PDF and CSV audits via ReportLab.
- **Alert Dispatcher**: Automated notifications on capacity threshold or hazard detection.
- **Containerization**: Production Dockerfile and docker-compose configurations included.

</td>
</tr>
</table>

---

## Technology Stack

| Layer | Component | Description |
|---|---|---|
| **Computer Vision** | Ultralytics YOLOv8 | Edge object detection and material categorization |
| **Edge LLM** | Llama 3.2 1B Instruct | Quantized on-device reasoning (GGUF / ExecuTorch) |
| **Application Server** | Flask 3.1 | REST API, SSE event streaming, and telemetry server |
| **Desktop Runtime** | OpenCV 4.x | Direct V4L2 camera capture and HUD overlays |
| **Database** | SQLite 3 | On-device persistent event and sensor logging |
| **Firmware** | MicroPython | Sensor polling and serial protocol on Raspberry Pi Pico W |
| **Containerization** | Docker / Compose | Multi-stage container deployments for cloud and edge |
| **Hardware Targets** | Arm Cortex-A / Zynq-7000 | Raspberry Pi 5, ZedBoard, ZC702, and Hailo-8 NPU |

---

## Quick Start

### 1. Prerequisites
- Linux (Ubuntu 22.04+, Debian Bookworm, or Raspberry Pi OS) / macOS / Windows WSL2
- Python 3.11 or 3.12
- Git

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/keer9999/bharath-ai-soc.git
cd bharath-ai-soc

# Run automated setup script
bash scripts/setup.sh --dev

# Download quantized model assets (Llama 3.2 1B GGUF)
bash scripts/download_models.sh
```

### 3. Launching Applications

#### Flask Web Interface
```bash
source .venv/bin/activate
PYTHONPATH=src python src/web_app.py --host 0.0.0.0 --port 5000
# Accessible at: http://localhost:5000
```

#### OpenCV Desktop HUD Console
```bash
source .venv/bin/activate
PYTHONPATH=src python src/main.py
```

#### Docker Compose Deployment
```bash
docker compose up --build -d
# Web console available at http://localhost:5000
```

### 4. Running Test Suite

```bash
# Run all unit and regression tests
make test

# Or execute with coverage report
make test-cov
```

---

## Hardware Setup

For complete wiring diagrams, pinout mappings, and hardware schematics, refer to **[docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md)**.

### Sensor Pin Map (Raspberry Pi Pico W)

| Sensor | Sensor Pin | Pico W GPIO | Functional Description |
|---|---|---|---|
| **HC-SR04** | TRIG / ECHO | GP3 / GP2 | Ultrasonic bin fill depth measurement |
| **DHT11** | DATA | GP15 | Ambient temperature and humidity monitoring |
| **MQ-135** | AO / DO | GP26 (ADC0) / GP16 | Hazardous gas and decomposition detection |
| **HW-103** | AO | GP27 (ADC1) | Soil and organic matter moisture sensing |

---

## API Reference

The Flask application exposes a complete REST API. Detailed schemas are documented in **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)**.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System health, camera status, and connected model status |
| `POST` | `/api/detect` | Execute YOLOv8 detection on an uploaded image |
| `GET` | `/api/webcam/stream` | Continuous MJPEG live video stream with HUD bounding boxes |
| `POST` | `/api/chat` | Send queries to Llama 3.2 / WasteAdvisor for disposal advice |
| `GET` | `/api/stats` | Aggregated waste statistics, category counts, and fill metrics |
| `GET` | `/api/export/pdf` | Generate and download an audit report in PDF format |
| `GET` | `/api/export/csv` | Export telemetry detection events in CSV format |

---

## Model Training and Optimization

Custom YOLOv8 models are trained on domain-specific datasets (TrashNet, TACO, and WasteNet):

```bash
PYTHONPATH=src python src/train_waste_model.py \
  --data data/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640
```

Refer to **[docs/TRAINING.md](docs/TRAINING.md)** for data preparation guidelines, synthetic augmentation pipelines, and ONNX/Hailo compilation instructions.

---

## Repository Structure

```text
BHARATH-AI-SOC/
├── .github/                   # GitHub Actions CI/CD workflows and issue templates
│   ├── workflows/ci.yml       # Linting, testing, and container build checks
│   └── workflows/deploy.yml   # Embedded SSH deployment automation
├── .devcontainer/             # GitHub Codespaces 1-click cloud development environment
├── src/                       # Production application source code
│   ├── web_app.py             # Flask Web Server & REST API endpoints
│   ├── main.py                # OpenCV Desktop HUD Console application
│   ├── waste_detector.py      # Multi-model YOLOv8 detection engine
│   ├── waste_advisor.py       # Domain knowledge engine for recycling recommendations
│   ├── llama_engine.py        # Quantized Llama 3.2 1B on-device inference
│   ├── executorch_llama_engine.py # ExecuTorch Arm target runtime
│   ├── data_fusion.py         # Multi-sensor telemetry integration
│   ├── database.py            # SQLite ORM and telemetry data access layer
│   ├── alert_system.py        # Safety and environmental notification dispatch
│   ├── report_generator.py    # Automated PDF & CSV compliance reports
│   ├── webcam_capture.py      # Camera capture and frame buffering service
│   ├── config.py              # Centralized configuration schema
│   ├── web/                   # Responsive web templates, static assets, and UI
│   ├── pico_sensors/          # MicroPython firmware for Raspberry Pi Pico W
│   └── yolo/                  # Training pipelines and Colab master notebooks
├── hardware/                  # Embedded SoC and accelerator targets
│   ├── zynq7000/              # Xilinx Zynq-7000 / ZedBoard (BRAM + Vivado)
│   ├── zc702/                 # Xilinx ZC702 Development Board project
│   └── hailo_npu/             # Hailo-8 / Hailo-8L NPU acceleration integration
├── tests/                     # Automated test suite (taxonomy, web hardening, advisor)
├── docs/                      # Architectural specifications and user guides
│   ├── ARCHITECTURE.md        # Deep-dive system architecture specifications
│   ├── HARDWARE_SETUP.md      # Pin mappings, electrical connections, and assembly
│   ├── API_REFERENCE.md       # Complete REST API specification
│   ├── DEPLOYMENT.md          # Multi-target deployment procedures
│   └── TRAINING.md            # Vision model training and tuning guide
├── scripts/                   # System automation scripts
│   ├── setup.sh               # Environment provisioning script
│   ├── download_models.sh     # Model downloader for on-device weights
│   └── run_tests.sh           # Test harness script with coverage analysis
├── data/                      # Dataset descriptors and sample configurations
├── Dockerfile                 # Multi-stage production container image
├── docker-compose.yml         # Containerized development and service stack
├── Makefile                   # Build, test, and lifecycle targets
├── pyproject.toml             # Python package configuration and tool settings
├── requirements.txt           # Pinned production dependencies
├── CHANGELOG.md               # Version release history
├── CONTRIBUTING.md            # Contribution protocols and pull request checklist
├── CODE_OF_CONDUCT.md         # Contributor Covenant standard
├── SECURITY.md                # Vulnerability reporting protocol
└── LICENSE                    # MIT Open Source License
```

---

## Deployment Options

1. **Embedded Linux (Raspberry Pi 5)**: Native installation managed via systemd. Detailed in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.
2. **Containerized (Docker)**: Deploy using `docker compose up -d` for portable cloud or edge execution.
3. **Cloud Preview (GitHub Codespaces)**: Launch directly in the browser via Codespaces with pre-configured Python 3.11, extensions, and port forwarding.
4. **FPGA SoC (Xilinx Zynq-7000 / ZedBoard)**: PetaLinux deployment with BRAM direct memory access for hardware acceleration.

---

## Contributing

Contributions are welcome. Please consult **[CONTRIBUTING.md](CONTRIBUTING.md)** and **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** for style standards, commit conventions, and review procedures.

```bash
git checkout -b feat/your-feature-name
git commit -m "feat: implement feature description"
git push origin feat/your-feature-name
```

---

## License

This project is licensed under the [MIT License](LICENSE).
