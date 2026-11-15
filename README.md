<div align="center">

# 🗑️ Bharat AI-SoC Challenge 2026-27 — PS2
## AI-Powered Waste Classification & Disposal Assistant

[![CI](https://github.com/vskeerthan/c2s/actions/workflows/ci.yml/badge.svg)](https://github.com/vskeerthan/c2s/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20%7C%20Zynq--7000-red?logo=raspberrypi&logoColor=white)](docs/HARDWARE_SETUP.md)
[![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-00FFFF?logo=ultralytics)](https://docs.ultralytics.com/)
[![Llama](https://img.shields.io/badge/LLM-Llama%203.2%201B-6C2FBF)](https://ai.meta.com/llama/)
[![Arm](https://img.shields.io/badge/Powered%20by-Arm-0091BD?logo=arm&logoColor=white)](https://www.arm.com/)
[![MeitY](https://img.shields.io/badge/Organised%20by-MeitY%20C2S-FF6B00)](https://www.meity.gov.in/)

<br/>

> **A real-time, edge-AI waste management system** that runs entirely on-device — combining YOLOv8 computer vision, Llama 3.2 1B language guidance, multi-sensor fusion, and an FPGA accelerator — to classify, track, and advise on proper waste disposal.

<br/>

**[📖 Documentation](docs/)** · **[🏗️ Architecture](docs/ARCHITECTURE.md)** · **[🔧 Hardware Setup](docs/HARDWARE_SETUP.md)** · **[🚀 Deployment](docs/DEPLOYMENT.md)** · **[🤝 Contributing](CONTRIBUTING.md)**

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏛️ System Architecture](#️-system-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚡ Quick Start](#-quick-start)
- [🔩 Hardware Setup](#-hardware-setup)
- [📡 API Reference](#-api-reference)
- [🧠 Model Training](#-model-training)
- [🏆 Competition Context](#-competition-context)
- [📁 Repository Structure](#-repository-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Intelligence
- **Real-time waste detection** at ≥15 FPS on RPi 5 using YOLOv8n/YOLOv8s
- **5-class classification**: Plastic · Glass · Metal · Paper · Organic
- **On-device LLM** (Llama 3.2 1B via llama-cpp-python GGUF) for disposal guidance in English & Hindi
- **Confidence-based alerts** with configurable thresholds

</td>
<td width="50%">

### 🔬 Sensor Fusion
- **Ultrasonic** (HC-SR04): Bin fill-level monitoring
- **DHT11**: Temperature & humidity for fermentation detection
- **MQ-135 Gas Sensor**: Methane/VOC detection for organic waste
- **Soil Moisture Sensor**: Compostability assessment
- Raspberry Pi Pico W orchestrates all sensors via UART/USB

</td>
</tr>
<tr>
<td width="50%">

### 🌐 Web Interface
- Live **MJPEG camera stream** with bounding-box overlays
- **Real-time chat** powered by Llama 3.2 1B
- **Analytics dashboard**: waste composition charts, trends, heatmaps
- PDF & CSV **report generation** on demand

</td>
<td width="50%">

### 🖥️ Desktop Application
- **OpenCV HUD overlays**: class labels, confidence scores, sensor readings
- Offline operation — no internet required
- Low-latency detection pipeline
- System tray integration with alert pop-ups

</td>
</tr>
<tr>
<td width="50%">

### 🔮 FPGA Acceleration (Zynq-7000)
- Hardware-accelerated inference via AXI DMA
- BRAM-based frame buffer for zero-copy transfer
- PS (ARM Cortex-A9) ↔ PL (FPGA fabric) co-design
- ZedBoard & ZC702 target support

</td>
<td width="50%">

### 📊 Data & Reports
- **SQLite** persistent detection history
- **PDF reports** (ReportLab) with charts & summaries
- **CSV export** for further analysis
- JSON API for third-party dashboard integration

</td>
</tr>
</table>

---

## 🏛️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BHARAT AI-SoC WASTE MANAGEMENT SYSTEM                 │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │                         SENSING LAYER                                 │
  │                                                                       │
  │   ┌─────────────────┐          ┌────────────────────────────────┐    │
  │   │  USB/MJPEG Cam  │          │     Raspberry Pi Pico W        │    │
  │   │  (1080p/720p)   │          │  ┌──────────┐ ┌────────────┐  │    │
  │   └────────┬────────┘          │  │ HC-SR04  │ │   DHT11    │  │    │
  │            │                   │  │Ultrasonic│ │ Temp/Hum   │  │    │
  │            │                   │  └──────────┘ └────────────┘  │    │
  │            │                   │  ┌──────────┐ ┌────────────┐  │    │
  │            │                   │  │  MQ-135  │ │  Moisture  │  │    │
  │            │                   │  │ Gas Snsr │ │   Sensor   │  │    │
  │            │                   │  └──────────┘ └────────────┘  │    │
  │            │                   └────────────┬───────────────────┘    │
  └────────────┼─────────────────────────────────┼──────────────────────┘
               │  OpenCV frames                   │  UART/USB JSON
  ┌────────────▼─────────────────────────────────▼──────────────────────┐
  │                      PROCESSING LAYER (RPi 5)                        │
  │                                                                       │
  │   ┌───────────────────┐    ┌────────────────────────────────────┐   │
  │   │   YOLOv8 Engine   │    │        Data Fusion Module          │   │
  │   │  waste_detector   │◄──►│  Fuses vision + sensor readings    │   │
  │   │  ≥15 FPS on RPi5  │    │  Calculates fill-level, risk score │   │
  │   └─────────┬─────────┘    └────────────────────────────────────┘   │
  │             │                                                         │
  │   ┌─────────▼─────────┐    ┌────────────────────────────────────┐   │
  │   │   Llama 3.2 1B    │    │         SQLite Database            │   │
  │   │  GGUF / ExecuTorch│    │  Detections, sessions, analytics   │   │
  │   │  Disposal guidance│    └────────────────────────────────────┘   │
  │   └─────────┬─────────┘                                              │
  │             │                                                         │
  │   ┌─────────▼─────────────────────────────────────────────────────┐ │
  │   │                  Alert & Report System                          │ │
  │   │          Email / SMS alerts · PDF/CSV report generation         │ │
  │   └─────────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────────┘
               │  FPGA offload (optional)         │  Flask API
  ┌────────────▼──────────────────┐  ┌────────────▼──────────────────┐
  │   ACCELERATION LAYER          │  │     PRESENTATION LAYER         │
  │   Zynq-7000 / ZedBoard/ZC702  │  │                                │
  │                               │  │  ┌──────────────────────────┐ │
  │  PS (ARM A9) ◄──AXI──► PL    │  │  │  Flask Web App           │ │
  │  BRAM frame buffer            │  │  │  MJPEG stream + Chat     │ │
  │  HW inference accelerator     │  │  │  Analytics dashboard     │ │
  └───────────────────────────────┘  │  └──────────────────────────┘ │
                                      │  ┌──────────────────────────┐ │
                                      │  │  OpenCV Desktop App      │ │
                                      │  │  HUD overlays + alerts   │ │
                                      │  └──────────────────────────┘ │
                                      └───────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Vision AI** | [YOLOv8](https://docs.ultralytics.com/) (Ultralytics) | Real-time waste object detection |
| **Language AI** | [Llama 3.2 1B](https://ai.meta.com/llama/) via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | On-device disposal guidance chatbot |
| **Edge Runtime** | [ExecuTorch](https://pytorch.org/executorch/) | Optimised ARM inference target |
| **Sensor MCU** | Raspberry Pi Pico W (MicroPython) | Multi-sensor orchestration |
| **Main Compute** | Raspberry Pi 5 (8GB) | Vision + LLM inference host |
| **FPGA** | Xilinx Zynq-7000 (ZedBoard / ZC702) | Hardware accelerated inference |
| **Web Framework** | Flask 3.x | REST API + MJPEG streaming |
| **Desktop** | OpenCV 4.x | HUD overlay desktop app |
| **Database** | SQLite 3 | Local detection history |
| **Reporting** | ReportLab + Matplotlib | PDF/CSV analytics reports |
| **Dashboard** | Streamlit + Plotly | Interactive analytics |
| **CI/CD** | GitHub Actions | Automated test & lint |
| **Container** | Docker + Compose | Reproducible dev environment |

---

## ⚡ Quick Start

### Prerequisites

- Python 3.11+
- OpenCV-compatible USB camera
- *(Optional)* Raspberry Pi Pico W with sensors
- *(Optional)* 4GB+ RAM recommended for Llama inference

### 1. Clone & Setup

```bash
git clone https://github.com/vskeerthan/bharat-aisoc-ps2.git
cd bharat-aisoc-ps2

# Automated setup (creates venv, installs deps, downloads model stub)
bash scripts/setup.sh
```

### 2. Download Models

```bash
# Download Llama 3.2 1B GGUF (requires ~800MB free space)
bash scripts/download_models.sh
```

### 3. Run the Web Application

```bash
source .venv/bin/activate
PYTHONPATH=src python src/web_app.py
# → Open http://localhost:5000
```

### 4. Run the Desktop Application

```bash
source .venv/bin/activate
PYTHONPATH=src python src/main.py
```

### 5. Run with Docker

```bash
docker compose up --build
# → Web interface at http://localhost:5000
```

### 6. Run Tests

```bash
make test
# Or using pytest directly:
PYTHONPATH=src pytest tests/ -v
# or
bash scripts/run_tests.sh
```

---

## 🔩 Hardware Setup

### Minimum Setup (Software Only)
| Component | Specification |
|-----------|--------------|
| Computer | Any x86-64 / ARM64 with 4GB RAM |
| Camera | Any USB webcam (OpenCV-compatible) |
| OS | Ubuntu 22.04 LTS / Raspberry Pi OS Bookworm |

### Full Edge Setup
| Component | Model | Purpose |
|-----------|-------|---------|
| Main SBC | Raspberry Pi 5 (8GB) | Vision + LLM inference |
| Sensor MCU | Raspberry Pi Pico W | Sensor data aggregation |
| Camera | USB / CSI 1080p | Waste image capture |
| Ultrasonic | HC-SR04 | Bin fill-level |
| Temp/Humidity | DHT11 | Environment monitoring |
| Gas Sensor | MQ-135 | VOC / methane detection |
| Moisture | Capacitive soil sensor | Compostability detection |
| FPGA (optional) | ZedBoard / ZC702 (Zynq-7000) | Hardware acceleration |

For detailed wiring diagrams and pin assignments, see **[📖 Hardware Setup Guide](docs/HARDWARE_SETUP.md)**.

---

## 📡 API Reference

The Flask web application exposes a REST + streaming API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard |
| `GET` | `/video_feed` | MJPEG camera stream |
| `GET` | `/api/status` | System health & sensor readings |
| `GET` | `/api/detections` | Recent detection history (JSON) |
| `GET` | `/api/stats` | Waste composition statistics |
| `POST` | `/api/chat` | Chat with Llama 3.2 1B assistant |
| `GET` | `/api/report/pdf` | Generate & download PDF report |
| `GET` | `/api/report/csv` | Export detection data as CSV |
| `POST` | `/api/alert/config` | Configure alert thresholds |

For full API documentation with request/response schemas, see **[📡 API Reference](docs/API_REFERENCE.md)**.

---

## 🧠 Model Training

The YOLOv8 model was trained on a custom waste dataset:

```bash
# Training from scratch
PYTHONPATH=src python src/train_waste_model.py \
  --data data/data.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640
```

For the full training guide including dataset preparation, augmentation strategies, and export options, see **[🧠 Training Guide](docs/TRAINING.md)**.

**Model Performance:**

| Model | mAP@0.5 | Inference (RPi 5) | Size |
|-------|---------|-------------------|------|
| YOLOv8n | ~72% | ~18 FPS | 6.3 MB |
| YOLOv8s | ~78% | ~12 FPS | 22 MB |

---

## 🏆 Competition Context

This project is submitted to the **[Bharat AI-SoC Challenge 2026-27](https://c2s.gov.in/)**, Problem Statement 2 (PS2).

| Field | Details |
|-------|---------|
| **Challenge** | Bharat AI System-on-Chip (AI-SoC) Challenge 2026-27 |
| **Problem Statement** | PS2 — AI-Powered Waste Classification & Disposal Assistant |
| **Organisers** | Arm · MeitY Chips to Startup (C2S) Programme · IIT Delhi |
| **Objective** | Build an edge AI system for smart waste management |
| **Target Hardware** | Arm Cortex-A (RPi 5) + Xilinx Zynq-7000 SoC |
| **Key Innovation** | Full on-device inference — no cloud dependency |

### Why Edge AI for Waste Management?

- 🔒 **Privacy**: No waste images leave the device
- ⚡ **Low Latency**: Sub-100ms detection response
- 🌐 **Offline Operation**: Works without internet in rural/industrial sites
- 🔋 **Energy Efficient**: Optimized for Arm & FPGA hardware
- 🇮🇳 **Multilingual**: Hindi + English disposal guidance

---

## 📁 Repository Structure

```
BHARATH-AI-SOC/
├── 📁 .github/                ← GitHub Actions CI/CD & templates
│   ├── workflows/
│   │   ├── ci.yml             ← Automated testing, linting & Docker build check
│   │   └── deploy.yml         ← RPi 5 SSH deployment workflow
│   └── ISSUE_TEMPLATE/        ← Bug & feature request templates
├── 📁 src/                    ← Core application package
│   ├── web_app.py             ← Flask REST API & Web Dashboard
│   ├── main.py                ← Desktop OpenCV HUD Application
│   ├── waste_detector.py      ← Dual-model YOLOv8 detection engine
│   ├── waste_advisor.py       ← 2,300+ line rule-based knowledge engine
│   ├── llama_engine.py        ← On-device Llama 3.2 1B inference (GGUF)
│   ├── executorch_llama_engine.py ← ExecuTorch on-device runtime
│   ├── data_fusion.py         ← Multi-sensor data fusion engine
│   ├── database.py            ← SQLite database ORM & analytics
│   ├── alert_system.py        ← Email / SMS hazard notification system
│   ├── report_generator.py    ← PDF & CSV automated report generator
│   ├── webcam_capture.py      ← Resilient camera capture abstraction
│   ├── config.py              ← Central configuration management
│   ├── web/                   ← Modern responsive web UI
│   │   ├── templates/         ← HTML5 templates (Jinja2)
│   │   └── static/            ← CSS, Vanilla JS, images, icons
│   ├── pico_sensors/          ← MicroPython firmware for Raspberry Pi Pico W
│   ├── yolo/                  ← Training scripts, colab notebooks & weights
│   └── models/                ← Model directory (Llama 3.2 GGUF & YOLO weights)
├── 📁 hardware/               ← SoC & Hardware Accelerator Targets
│   ├── zynq7000/              ← Xilinx Zynq-7000 / ZedBoard SoC (BRAM + Vivado)
│   ├── zc702/                 ← Xilinx ZC702 Development Board Vivado project
│   └── hailo_npu/             ← Hailo-8 / Hailo-8L NPU acceleration integration
├── 📁 tests/                  ← Pytest test suite (mapping, taxonomy, hardening)
├── 📁 docs/                   ← Comprehensive system documentation
│   ├── ARCHITECTURE.md        ← Detailed system architecture & data flows
│   ├── HARDWARE_SETUP.md      ← Sensor wiring & hardware configuration
│   ├── API_REFERENCE.md       ← REST API endpoint documentation
│   ├── DEPLOYMENT.md          ← Deployment guide (RPi 5, Docker, FPGA)
│   ├── TRAINING.md            ← YOLOv8 custom dataset training guide
│   ├── PROJECT_REPORT.md      ← Technical project audit report
│   └── SoC_Migration_Guide.docx ← SoC migration whitepaper
├── 📁 scripts/                ← Utility & automation scripts
│   ├── setup.sh               ← Automated environment setup script
│   ├── download_models.sh     ← Llama 3.2 1B GGUF model downloader
│   └── run_tests.sh           ← Test suite runner with coverage
├── 📁 data/                   ← Dataset samples & configurations
├── 🐳 Dockerfile              ← Production multi-stage container
├── 🐳 docker-compose.yml      ← Local development stack definition
├── 🛠️ Makefile                ← Development automation (test, run, lint)
├── 📦 pyproject.toml          ← Project metadata, packaging & tool config
├── 📄 requirements.txt        ← Frozen Python dependencies
├── 📜 LICENSE                 ← MIT Open Source License
├── 📝 CHANGELOG.md            ← Version history & release notes
├── 🤝 CONTRIBUTING.md         ← Contribution guidelines & style guide
├── 🛡️ CODE_OF_CONDUCT.md      ← Contributor Covenant v2.1
└── 🔒 SECURITY.md             ← Vulnerability disclosure policy
```

---

## 🤝 Contributing

Contributions are welcome! Please read our **[Contributing Guide](CONTRIBUTING.md)** and **[Code of Conduct](CODE_OF_CONDUCT.md)** before getting started.

```bash
# Fork → Clone → Branch → Commit → PR
git checkout -b feat/your-feature-name
git commit -m "feat: add awesome feature"
git push origin feat/your-feature-name
```

---

## 🔒 Security

Found a vulnerability? Please follow our **[Security Policy](SECURITY.md)** for responsible disclosure.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ for Bharat · Powered by Arm · Supported by MeitY C2S**

*Making India's waste management smarter, one edge device at a time.*

</div>

