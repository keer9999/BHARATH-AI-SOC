# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- ExecuTorch export pipeline for Arm Cortex-A optimised inference
- Hindi language support for Llama disposal guidance
- Streamlit analytics dashboard (v2)
- MQTT integration for IoT gateway publishing
- Docker multi-arch build (arm64 for RPi 5)

---

## [1.0.0] — 2026-11-14

### Added — Bharat AI-SoC Challenge 2026-27 Initial Submission

#### Vision AI
- YOLOv8n/YOLOv8s real-time waste detection at ≥15 FPS on Raspberry Pi 5
- 5-class waste taxonomy: Plastic · Glass · Metal · Paper · Organic
- Confidence-based detection filtering with configurable thresholds
- Custom YOLOv8 training pipeline (`train_waste_model.py`)

#### On-Device LLM
- Llama 3.2 1B GGUF inference via `llama-cpp-python`
- ExecuTorch engine stub for future Arm-optimised deployment (`executorch_llama_engine.py`)
- Context-aware disposal guidance per waste class
- Interactive chat interface in web and desktop apps

#### Multi-Sensor Fusion
- Raspberry Pi Pico W MicroPython firmware for 4-sensor array
- HC-SR04 ultrasonic: bin fill-level monitoring with overfill alerts
- DHT11: temperature & humidity for fermentation/organic detection
- MQ-135 gas sensor: methane/VOC detection
- Capacitive moisture sensor: compostability assessment
- Real-time sensor data fusion with vision detections (`data_fusion.py`)
- UART/USB JSON protocol for Pico W → RPi 5 communication

#### Web Application
- Flask 3.x web server with MJPEG camera streaming
- Real-time waste classification overlay on live feed
- Llama-powered chat assistant endpoint (`/api/chat`)
- RESTful API: `/api/status`, `/api/detections`, `/api/stats`
- WebSocket-ready architecture for future real-time push

#### Desktop Application
- OpenCV HUD overlays with class labels and confidence scores
- Sensor readings panel in real-time
- Keyboard shortcuts for common operations
- Offline-first architecture

#### Database & Reporting
- SQLite detection history with full session tracking
- PDF report generation (ReportLab + Matplotlib charts)
- CSV export for external analysis
- JSON report format for API consumers

#### Alerts
- Email alert system for configurable waste accumulation thresholds
- Bin overfill alerts via ultrasonic sensor
- Gas hazard alerts (MQ-135) with severity levels

#### FPGA Integration (Zynq-7000)
- Vivado project structure for ZedBoard and ZC702 boards
- AXI DMA configuration for PS ↔ PL frame transfer
- BRAM-based frame buffer implementation
- ARM Cortex-A9 (PS) ↔ FPGA fabric (PL) co-design guide

#### Developer Experience
- Comprehensive `.gitignore` for Python, ML, Node, IDE, OS
- GitHub Actions CI: pytest + ruff linting + build check
- GitHub Actions deploy workflow: SSH-based RPi 5 deployment
- Issue templates: bug report + feature request
- Pull request template
- `pyproject.toml` with ruff, pytest, and project metadata
- `Dockerfile` and `docker-compose.yml` for containerised dev
- `Makefile` with install, run, test, lint, docker-build, clean targets
- `scripts/setup.sh` automated environment setup
- `scripts/download_models.sh` model acquisition helper

#### Documentation
- World-class `README.md` with ASCII architecture diagram, badges, quick-start
- `docs/ARCHITECTURE.md` — full system design with ASCII diagrams
- `docs/HARDWARE_SETUP.md` — Pico W wiring, RPi 5 setup, Zynq-7000 guide
- `docs/API_REFERENCE.md` — complete Flask API documentation
- `docs/DEPLOYMENT.md` — RPi 5, Zynq-7000, and Docker deployment guides
- `docs/TRAINING.md` — YOLOv8 training guide with dataset prep
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`

---

[Unreleased]: https://github.com/vskeerthan/bharat-aisoc-ps2/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/vskeerthan/bharat-aisoc-ps2/releases/tag/v1.0.0

