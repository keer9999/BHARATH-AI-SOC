<div align="center">

<img src="https://img.shields.io/badge/Bharat%20AI--SoC-2026--27-FF6B00?style=for-the-badge&logo=arm&logoColor=white" alt="Bharat AI-SoC 2026-27"/>

# EcoSort AI
### On-Device Multimodal Waste Perception, Slicing Inference & Circular Recycling Platform

**Problem Statement 2 — AI-Powered Edge Waste Classification & Disposal Assistant**  
*Organised by Arm · MeitY C2S Programme · IIT Delhi*

<br/>

<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white" alt="Python 3.11+"/></a>
<a href="src/yolo/weights/"><img src="https://img.shields.io/badge/Vision-YOLO11s_%C2%B7_66_Classes_%C2%B7_ONNX-00C853?logo=opencv&logoColor=white" alt="YOLOv11"/></a>
<a href="src/executorch_llama_engine.py"><img src="https://img.shields.io/badge/LLM-Llama_3.2_1B_%C2%B7_ExecuTorch-6C2FBF?logo=meta&logoColor=white" alt="LLM"/></a>
<a href="hardware/"><img src="https://img.shields.io/badge/Hardware-RPi_4B_%C2%B7_ZedBoard_%C2%B7_Zynq--7000-0091BD?logo=arm&logoColor=white" alt="Hardware"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
<a href="tests/"><img src="https://img.shields.io/badge/Tests-58%20Passing-brightgreen" alt="Tests"/></a>

</div>

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [Two Hardware Approaches](#two-hardware-approaches)
  - [Approach A — Zynq-7000 (ZedBoard) + Raspberry Pi 4B/5](#approach-a--zynq-7000-zedboard--raspberry-pi-4b5)
  - [Approach B — Raspberry Pi Only](#approach-b--raspberry-pi-only)
- [AI Model Architecture](#ai-model-architecture)
- [On-Device Language Model](#on-device-language-model)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Raspberry Pi Deployment (Keerthan's Pi)](#raspberry-pi-deployment-keerthans-pi)
- [Zynq-7000 / ZedBoard Deployment](#zynq-7000--zedboard-deployment)
- [Running the Test Suite](#running-the-test-suite)
- [Technical Report (LaTeX)](#technical-report-latex)
- [Competition Alignment](#competition-alignment)
- [Technology Stack](#technology-stack)
- [Model Weights & Files](#model-weights--files)

---

## What This Project Does

**EcoSort AI** is a fully on-device, cloud-free waste classification and disposal guidance system built for the **Bharat AI-SoC Challenge 2026-27 (PS-2)**. It runs the complete AI pipeline — vision detection, language reasoning, sensor fusion, and web dashboard — locally on ARM-based edge hardware.

### Core Capabilities

| Capability | Technology | Detail |
|-----------|------------|--------|
| **66-Class Waste Detection** | YOLO11s ONNX | Single unified model: plastics, glass, metal, paper, organic, e-waste, hazardous |
| **Small-Object SAHI** | Pure-NumPy WBF | Slicing Aided Hyper Inference tiles the frame to catch small items |
| **On-Device LLM** | Llama 3.2 1B | Disposal advice, upcycling recipes, resin code analysis — zero cloud |
| **Expert Rule Engine** | WasteAdvisor (2,300 rules) | Sub-5 ms deterministic fallback for all waste categories |
| **IoT Sensor Fusion** | Raspberry Pi Pico W | Bin fill level (HC-SR04), humidity (DHT11), gas (MQ-135), moisture |
| **Hardware Actuation** | Zynq-7000 AXI GPIO | Physical bin routing via onboard LEDs LD1–LD7 and relay triggers |
| **Live Web Dashboard** | Flask + MJPEG | Real-time webcam stream, detection HUD, analytics, PDF reports |

---

## Two Hardware Approaches

> 📖 **Comprehensive Engineering Comparison:** For the in-depth architectural trade-off matrix, power/latency profiling, and competition rubric justification, see [**docs/HARDWARE_APPROACHES.md**](docs/HARDWARE_APPROACHES.md).

### Approach A — Zynq-7000 (ZedBoard) + Raspberry Pi 4B/5

> **Recommended for the full competition submission** — demonstrates PS+PL co-design on an Arm SoC as specifically required by the challenge.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     APPROACH A — FULL SoC STACK                         │
│                                                                          │
│  Raspberry Pi Pico W          Raspberry Pi 4B / 5                       │
│  (MicroPython sensors)        (AI Processing Node)                      │
│  HC-SR04 · DHT11 ─── USB ──► waste_detector.py (YOLO11s ONNX)         │
│  MQ-135 · Moisture           llama_engine.py (Llama 3.2 1B GGUF)       │
│                               sahi_detector.py (tiled inference)        │
│                               web_app.py ──► HTTP:5001 ──► Browser      │
│                                     │                                    │
│                               Serial / SPI                               │
│                                     │                                    │
│                      ┌──────────────▼───────────────┐                   │
│                      │   Xilinx Zynq-7000 ZedBoard   │                   │
│                      │   Dual-core ARM Cortex-A9      │                   │
│                      │   + Programmable Logic (PL)    │                   │
│                      │                                │                   │
│                      │   AXI GPIO → LD1-LD7 LEDs     │                   │
│                      │   BRAM-mapped bin routing      │                   │
│                      │   PetaLinux 2024.1             │                   │
│                      └────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why use the Zynq?**
- The PS (Processing System, ARM Cortex-A9) runs PetaLinux + Python + the full web stack.
- The PL (Programmable Logic, FPGA fabric) handles AXI GPIO for deterministic bin actuation in <1 µs latency.
- This directly satisfies the challenge requirement: *"deployment on an Arm-based System-on-Chip with FPGA co-processing."*
- The Zynq board communicates with the Raspberry Pi inference node over serial/SPI, offloading the heavy AI compute to the more capable Pi while using the Zynq for deterministic hardware control.

**→ Full guide:** [`hardware/zynq7000/README_ZYNQ7000.md`](hardware/zynq7000/README_ZYNQ7000.md)

---

### Approach B — Raspberry Pi Only

> **Simpler setup** — runs the entire stack on a single Raspberry Pi 4B or 5. No FPGA required.

```
┌────────────────────────────────────────────────────────────────────┐
│               APPROACH B — RASPBERRY PI STANDALONE                  │
│                                                                     │
│  Raspberry Pi Pico W             Raspberry Pi 4B / 5 (8 GB)        │
│  (Sensor node — MicroPython)     (Full AI + Web stack)             │
│                                                                     │
│  HC-SR04 ultrasonic ─── USB ──► waste_detector.py                  │
│  DHT11 temp/humidity             YOLO11s · 66 classes · ONNX       │
│  MQ-135 gas sensor               ↓                                 │
│  HW-103 moisture        llama_engine.py                             │
│                          Llama 3.2 1B GGUF (750 MB)                │
│                          OR ExecuTorch .pte (faster on Pi 5)       │
│                                  ↓                                 │
│                          Flask web_app.py → port 5001              │
│                          Accessible from any device on LAN         │
└────────────────────────────────────────────────────────────────────┘
```

**Why choose this?**
- Zero dependency on FPGA toolchain (no Vivado / PetaLinux required).
- Full system in a single \$80 board.
- Still fully cloud-free and on-device.
- Raspberry Pi 5 with the XNNPACK ExecuTorch backend matches or exceeds Jetson Nano performance on this model.

**→ Full guide:** [`raspberry_pi/keerthans_pi/README.md`](raspberry_pi/keerthans_pi/README.md)

---

## AI Model Architecture

### Unified 66-Class ONNX Model (`ecosort_waste_yolo11s.onnx`)

A single YOLO11s model trained on 66 waste categories replaces the previous 8-model specialist ensemble. This is the **recommended model for both Approach A and B**.

```
Input:  [1, 3, 640, 640]  float32
Output: [1, 70, 8400]     float32
Size:   37 MB
Opset:  ONNX 12
Speed:  ~300–500 ms/frame on Raspberry Pi 4B CPU
        ~150–200 ms on Raspberry Pi 5
        ~40–80 ms on Jetson Nano (GPU)
```

**All 66 classes by material group:**

| Group | Classes |
|-------|---------|
| **Plastic** | PET bottle, HDPE bottle, PVC pipe, LDPE film, PP container, PS foam, plastic bag, plastic cup, plastic straw, plastic cutlery |
| **Glass** | clear glass, brown glass, green glass, broken glass, glass jar, glass bottle |
| **Metal** | aluminium can, food can, steel scrap, copper wire, aerosol can |
| **Paper** | newspaper, cardboard box, cardboard flat, paper cup, paper bag, office paper |
| **Organic** | food waste, vegetable peel, fruit peel, coffee grounds, eggshell, leaves |
| **E-Waste** | phone, laptop, keyboard, remote, battery, circuit board, cable, light bulb |
| **Hazardous** | chemical bottle, paint can, motor oil container, medical waste, syringe |
| **Textile** | clothing, fabric scrap |

### SAHI — Slicing Aided Hyper Inference

For small or densely packed waste items (e.g. resin chips on a conveyor belt), EcoSort uses a **pure-NumPy SAHI engine** (`src/sahi_detector.py`) with Weighted Box Fusion (WBF). This requires no additional model and works with any ONNX backbone.

```
Frame 1280×720
    │
    ├─ Tile 0,0 (640×640) → YOLO detections
    ├─ Tile 0,1 (640×640) → YOLO detections
    ├─ Tile 1,0 (640×640) → YOLO detections
    └─ Tile 1,1 (640×640) → YOLO detections
                     │
              Coordinate remap
                     │
           Weighted Box Fusion (WBF)
                     │
           Fused, deduplicated detections
```

SAHI profiles available: `default`, `rpi4`, `rpi5`, `high_res`, `fast`

---

## On-Device Language Model

### Llama 3.2 1B — GGUF backend (default)

```bash
Model : Llama-3.2-1B-Instruct-Q4_K_M.gguf
Size  : ~750 MB
RAM   : ~700 MB at peak
Speed : 3–8 tokens/sec on Raspberry Pi 4B
        6–12 tokens/sec on Raspberry Pi 5
Path  : src/models/llama-3.2-1b-instruct-q4_k_m.gguf
```

Download it with:
```bash
bash scripts/download_models.sh
```

### ExecuTorch .pte backend (optional, recommended for Pi 5)

Meta's ExecuTorch ARM XNNPACK backend gives ~2× faster inference.  
Full setup: [`raspberry_pi/keerthans_pi/docs/EXECUTORCH_SETUP.md`](raspberry_pi/keerthans_pi/docs/EXECUTORCH_SETUP.md)

### WasteAdvisor — Deterministic fallback (always active)

When the LLM is not loaded (low RAM situations), `src/waste_advisor.py` provides instant (<5 ms) expert answers via 2,300 verified rules covering:
- Resin Identification Codes #1–#7 (RIC/SPI codes)
- Decomposition timelines for 200+ waste materials
- Bin routing by municipality type (dry/wet/e-waste/hazardous)
- DIY upcycling suggestions

---

## Repository Structure

```
BHARATH-AI-SOC/
│
├── src/                              # Main application source
│   ├── web_app.py                    # Flask web server & REST API
│   ├── waste_detector.py             # YOLO inference engine (ONNX + PT)
│   ├── sahi_detector.py              # SAHI tiled inference + WBF
│   ├── llama_engine.py               # Llama 3.2 GGUF runtime
│   ├── executorch_llama_engine.py    # ExecuTorch .pte runtime (ARM)
│   ├── waste_advisor.py              # 2,300-rule deterministic engine
│   ├── config.py                     # Centralised system configuration
│   ├── database.py                   # SQLite session & record management
│   ├── data_fusion.py                # Pico W sensor fusion
│   ├── report_generator.py           # PDF / CSV automated reports
│   ├── webcam_capture.py             # Thread-safe camera manager
│   ├── main.py                       # OpenCV desktop HUD app
│   ├── unified_detector.py           # Unified ONNX inference wrapper
│   ├── models/                       # Llama GGUF model (download via script)
│   └── yolo/
│       └── weights/
│           ├── ecosort_waste_yolo11s.onnx   ← unified 66-class (tracked)
│           ├── ecosort_waste_yolo11s.pt     ← PyTorch source
│           ├── ewaste_a.pt                  ← e-waste specialist (12 class)
│           ├── ewaste_b.pt                  ← e-waste specialist (77 class)
│           ├── food_organic.pt              ← food/organic (39 class)
│           ├── plastic_bag.pt               ← plastic bag (1 class)
│           ├── plastic_containers.pt        ← containers (3 class)
│           ├── resin_codes.pt               ← resin RIC codes (6 class)
│           ├── general_waste.pt             ← 5-class general
│           └── bottle_specialist.pt         ← bottle (1 class)
│
├── hardware/
│   ├── zynq7000/                     # Xilinx Zynq-7000 / ZedBoard
│   │   ├── README_ZYNQ7000.md        # Full deployment guide
│   │   ├── bram/                     # AXI GPIO + BRAM actuation code
│   │   │   ├── main_zynq.py          # Main loop (runs on PS ARM)
│   │   │   ├── bram_direct_io.py     # Memory-mapped GPIO driver
│   │   │   └── config_zynq.py        # Zynq-specific configuration
│   │   ├── vivado/                   # FPGA design files
│   │   │   ├── zedboard_block_design.tcl    # Vivado block design script
│   │   │   ├── zedboard_constraints.xdc     # I/O pin constraints
│   │   │   ├── deploy_to_zedboard.sh        # SD card flash script
│   │   │   ├── petalinux_recipe.bb          # PetaLinux app recipe
│   │   │   └── c2s-waste-zedboard.service   # systemd service
│   │   └── external/                 # External host interface (Pi→Zynq)
│   │       ├── web_app.py            # Web app variant for Zynq PS
│   │       ├── pico_sensors/main.py  # Pico W MicroPython firmware
│   │       └── models/               # Model placeholder (download separately)
│   │
│   ├── zc702/                        # Xilinx ZC702 evaluation board
│   │   ├── README_ZC702.md
│   │   ├── bram/                     # AXI-mapped actuation
│   │   └── vivado/                   # Vivado project files
│   │
│   └── hailo_npu/                    # Hailo-8 NPU integration (optional)
│       ├── hailo_export.py           # Export ONNX → HEF (Hailo format)
│       ├── inference_backend.py      # Hailo runtime inference
│       └── HAILO_INTEGRATION_GUIDE.md
│
├── raspberry_pi/
│   └── keerthans_pi/                 # Approach B — Pi-only deployment
│       ├── README.md                 # Full deployment guide
│       ├── requirements-pi.txt       # Pi-specific deps (onnxruntime)
│       ├── config/pi_config.env      # Pre-configured env vars
│       ├── scripts/
│       │   ├── setup_pi.sh           # One-shot setup
│       │   ├── start_pi.sh           # Start web app
│       │   └── download_llama.sh     # Download Llama GGUF
│       └── docs/EXECUTORCH_SETUP.md  # ExecuTorch ARM guide
│
├── tests/                            # Automated test suite (58 cases)
│   ├── test_sahi_integration.py
│   ├── test_v2_taxonomy.py
│   ├── test_web_hardening.py
│   ├── test_waste_advisor.py
│   ├── test_material_mapping.py
│   └── test_executorch_llama.py
│
├── docs/
│   ├── technical_report.tex          # Complete LaTeX Technical Report (PS-2 Submission)
│   ├── technical_report.bib          # BibTeX references database
│   ├── HARDWARE_APPROACHES.md        # Comprehensive Approach A vs B engineering matrix
│   ├── ARCHITECTURE.md               # System architecture diagrams
│   ├── API_REFERENCE.md              # REST endpoint specifications
│   ├── HARDWARE_SETUP.md             # Sensor wiring & board setup
│   ├── TRAINING.md                   # Model training logs
│   ├── DEPLOYMENT.md                 # Deployment procedures
│   └── PROJECT_REPORT.md             # Full engineering report
│
├── data/                             # Training datasets (yaml configs)
├── scripts/
│   ├── download_models.sh            # Download Llama GGUF weights
│   ├── build_unified_model.py        # Export YOLO11s → ONNX
│   ├── run_tests.sh                  # Run full test suite
│   └── setup.sh                      # Development environment setup
│
├── Dockerfile                        # Docker image (CPU, local dev)
├── docker-compose.yml                # Docker Compose for local testing
├── requirements.txt                  # Python dependencies
├── Makefile                          # Common dev tasks
└── README.md                         ← You are here
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- A USB webcam (optional — image upload works without camera)
- 2 GB RAM minimum (4 GB+ recommended)

### Method 1 — Local development (Linux / macOS / WSL)

```bash
# 1. Clone the repository
git clone https://github.com/keer9999/BHARATH-AI-SOC.git
cd BHARATH-AI-SOC

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download Llama 3.2 1B GGUF model (~750 MB)
bash scripts/download_models.sh

# 5. Run the web application
python src/web_app.py --port 5001

# Open: http://localhost:5001
```

### Method 2 — Docker (local testing)

```bash
docker compose up --build
# Open: http://localhost:5001
```

### Method 3 — CLI detection (no browser needed)

```bash
# Single image analysis
PYTHONPATH=src python src/main.py --image path/to/image.jpg

# Self-test with no camera or model needed
PYTHONPATH=src python src/main.py --self-test
```

---

## Raspberry Pi Deployment (Keerthan's Pi)

> **Approach B** — full AI stack on a single Raspberry Pi 4B or 5.

### Three-command setup

```bash
# On your Raspberry Pi:
git clone https://github.com/keer9999/BHARATH-AI-SOC.git
cd BHARATH-AI-SOC
bash raspberry_pi/keerthans_pi/scripts/setup_pi.sh
```

This script:
1. Installs system packages (`apt`)
2. Creates a Python virtualenv
3. Installs `onnxruntime`, `flask`, `llama-cpp-python`, `pyserial`
4. Downloads Llama 3.2 1B GGUF
5. Runs a smoke test

### Start the app

```bash
bash raspberry_pi/keerthans_pi/scripts/start_pi.sh
# Web dashboard: http://<pi-ip>:5001
```

### Auto-start on boot (systemd)

```bash
sudo tee /etc/systemd/system/ecosort.service > /dev/null <<EOF
[Unit]
Description=EcoSort AI
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/BHARATH-AI-SOC
ExecStart=/bin/bash /home/pi/BHARATH-AI-SOC/raspberry_pi/keerthans_pi/scripts/start_pi.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable --now ecosort
sudo journalctl -u ecosort -f
```

### Expected Pi performance

| Metric | Pi 4B (4 GB) | Pi 5 (8 GB) |
|--------|-------------|------------|
| ONNX model load | ~2 s | ~1 s |
| Detection per frame | 300–500 ms | 150–250 ms |
| Llama GGUF load | 10–15 s | 6–10 s |
| LLM tokens/sec (GGUF) | 3–6 tok/s | 6–12 tok/s |
| LLM tokens/sec (ExecuTorch) | 5–8 tok/s | 10–16 tok/s |
| Total RAM at runtime | ~900 MB | ~900 MB |
| Minimum recommended RAM | **4 GB** | **4 GB** |

---

## Zynq-7000 / ZedBoard Deployment

> **Approach A** — full PS+PL co-design on Xilinx Zynq-7000. See [`hardware/zynq7000/README_ZYNQ7000.md`](hardware/zynq7000/README_ZYNQ7000.md) for the complete guide.

### Quick overview

```bash
# 1. Build the Vivado project (run on a machine with Vivado 2024.1 installed)
cd hardware/zynq7000/vivado
vivado -mode batch -source zedboard_block_design.tcl

# 2. Flash SD card with PetaLinux image
bash deploy_to_zedboard.sh /dev/sdX

# 3. On the ZedBoard (after boot):
python hardware/zynq7000/bram/main_zynq.py --test   # Verify AXI GPIO + LEDs

# 4. Run continuous classification
python hardware/zynq7000/bram/main_zynq.py
```

### AXI GPIO LED Mapping

| LED | Waste Category | Bin |
|-----|---------------|-----|
| LD1 | Plastic | Blue bin |
| LD2 | Glass | Green bin |
| LD3 | Metal | Grey bin |
| LD4 | Paper / Cardboard | Yellow bin |
| LD5 | Organic / Food | Brown bin |
| LD6 | E-Waste | Purple bin |
| LD7 | Hazardous | Red bin |

---

## Running the Test Suite

All 58 tests run without a camera, GPU, or internet connection:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

| Test File | What it covers |
|-----------|---------------|
| `test_sahi_integration.py` | Tile coordinates, letterbox padding, WBF, NMS, SAHI profiles, REST endpoints |
| `test_v2_taxonomy.py` | 66-class taxonomy, legacy zero-regression, material mapping |
| `test_web_hardening.py` | 25 MB upload guard, HTTP 400 on bad input, concurrency locking |
| `test_waste_advisor.py` | Circular economy advice, resin codes, decomposition rates |
| `test_material_mapping.py` | Token normalization across edge cases |
| `test_executorch_llama.py` | Prompt assembly, ExecuTorch fallback paths |

---

## Technical Report (LaTeX)

The formal academic technical report submitted for **Bharat AI-SoC Challenge 2026–27 (Problem Statement 2: AI-Powered Edge Waste Classification and Disposal Assistant)** is authored in standard LaTeX:

- 📄 **LaTeX Source:** [`docs/technical_report.tex`](docs/technical_report.tex)
- 📚 **Bibliography Database:** [`docs/technical_report.bib`](docs/technical_report.bib)

### Key Sections Covered:
1. **Executive Summary & PS-2 Compliance:** Direct mapping against all mandatory and optional requirements.
2. **System Architecture:** High-level block diagram (TikZ) connecting camera, sensors, YOLO11s ONNX, ExecuTorch Llama, and Zynq-7000.
3. **Hardware Architectures:** Complete engineering evaluation of **Approach A** (Zynq-7000 + RPi co-design with $<1\,\mu$s AXI GPIO actuation) vs **Approach B** (Standalone RPi 5 monolithic deployment).
4. **Independent Pipelines:** Rigorous architectural separation of the **HailoRT NPU vision pipeline** and the **ExecuTorch ARM CPU LLM pipeline** as mandated by competition rules.
5. **On-Device Llama via ExecuTorch (Mandatory):** Model selection (Llama 3.2 1B Instruct), `.pte` export, XNNPACK ARM delegate, and token generation benchmarks.
6. **Deterministic Fallback:** 2,300-rule `WasteAdvisor` expert system ($<5$\,ms response, RIC #1–#7 resin codes).
7. **Empirical Benchmarks:** Latency, tokens/sec, and memory footprint tables across Raspberry Pi 4B, Raspberry Pi 5, and Hailo-8 AI HAT+.

### How to Compile the Report:

```bash
# Using pdflatex (single or double pass for TOC/references)
cd docs
pdflatex technical_report.tex
pdflatex technical_report.tex

# Using latexmk (automated multi-pass build)
latexmk -pdf technical_report.tex

# Or open docs/technical_report.tex directly in Overleaf (https://www.overleaf.com)
```

---

## Competition Alignment

| PS-2 Requirement | Implementation in EcoSort AI |
|-----------------|------------------------------|
| **Arm-based SoC target** | Xilinx Zynq-7000 (dual ARM Cortex-A9) + Raspberry Pi 4B/5 (ARM Cortex-A72/A76) |
| **On-device inference — no cloud** | Full ONNX + LLM pipeline runs locally; zero external API calls |
| **Computer Vision** | YOLO11s, 66 waste classes, SAHI tiled inference, Weighted Box Fusion |
| **Natural Language / Multimodal** | Llama 3.2 1B on-device (GGUF + ExecuTorch), WasteAdvisor 2,300 rules |
| **ExecuTorch (Arm requirement)** | `src/executorch_llama_engine.py` — full ExecuTorch runtime stub + guide |
| **Physical hardware actuation** | ZedBoard AXI GPIO → LD1–LD7 LEDs, relay trigger outputs |
| **IoT sensor integration** | Raspberry Pi Pico W (HC-SR04, DHT11, MQ-135, HW-103) over USB-CDC |
| **Web / Dashboard** | Flask 3.x, MJPEG live stream, REST API, analytics, PDF reports |
| **Reproducibility** | Docker, Makefile, automated tests, seed-locked exports |

---

## Technology Stack

| Domain | Library / Tool | Version |
|--------|---------------|---------|
| **Object Detection** | Ultralytics YOLO11 | ≥ 8.3.0 |
| **ONNX Inference** | onnxruntime-cpu | ≥ 1.17.0 |
| **On-Device LLM** | llama-cpp-python (GGUF) | ≥ 0.2.90 |
| **ARM LLM Runtime** | ExecuTorch XNNPACK | 0.3.x |
| **Computer Vision** | OpenCV (headless) | ≥ 4.8.0 |
| **Web Framework** | Flask | ≥ 3.0.0 |
| **FPGA Design** | Xilinx Vivado | 2024.1 |
| **Embedded OS** | PetaLinux | 2024.1 |
| **Sensor MCU** | Raspberry Pi Pico W (MicroPython) | 1.23+ |
| **Database** | SQLite 3 (built-in) | — |
| **Containerisation** | Docker + Compose | — |
| **Testing** | pytest | ≥ 7.0.0 |

---

## Model Weights & Files

All `.pt` model weights are tracked in the repository under `src/yolo/weights/`.  
The unified ONNX model is also tracked (37 MB):

| File | Size | Description |
|------|------|-------------|
| `ecosort_waste_yolo11s.onnx` | 37 MB | **Primary inference model** — 66 classes, ONNX 12, use on Pi / Zynq |
| `ecosort_waste_yolo11s.pt` | 19 MB | PyTorch source (use for further training) |
| `ewaste_a.pt` | 19 MB | E-waste specialist, 12 classes |
| `ewaste_b.pt` | 19 MB | E-waste specialist, 77 classes |
| `food_organic.pt` | 19 MB | Food & organic, 39 classes |
| `plastic_bag.pt` | 19 MB | Plastic bag specialist |
| `plastic_containers.pt` | 19 MB | Containers specialist |
| `resin_codes.pt` | 19 MB | Resin RIC code classifier |
| `general_waste.pt` | 19 MB | 5-class general model |
| `bottle_specialist.pt` | 20 MB | Bottle specialist |

> **Note:** Llama 3.2 1B GGUF (~750 MB) is **not** in the repository.  
> Download it with: `bash scripts/download_models.sh`

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

*Built for the Bharat AI-SoC Challenge 2026-27 · Problem Statement 2*  
*Arm · MeitY C2S · IIT Delhi*

</div>
