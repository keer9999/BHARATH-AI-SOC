<div align="center">

<img src="https://img.shields.io/badge/Bharat%20AI--SoC-2026--27-FF6B00?style=for-the-badge&logo=arm&logoColor=white" alt="Bharat AI-SoC"/>

# EcoSort AI
### On-Device Multimodal Waste Perception, Slicing Inference, and Circular Recycling Platform

*Problem Statement 2: AI-Powered Edge Waste Classification and Disposal Assistant*  
*Organized by Arm, MeitY C2S Programme, and IIT Delhi*

<br/>

[![CI](https://github.com/keer9999/BHARATH-AI-SOC/actions/workflows/ci.yml/badge.svg)](https://github.com/keer9999/BHARATH-AI-SOC/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Vision](https://img.shields.io/badge/Vision-YOLOv8%20%7C%20YOLO11%20%7C%20SAHI-00FFFF?logo=ultralytics&logoColor=black)](https://docs.ultralytics.com/)
[![LLM](https://img.shields.io/badge/LLM-Llama%203.2%201B-6C2FBF?logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![Hardware](https://img.shields.io/badge/SoC-Zynq--7000%20%7C%20ARM%20Cortex--A9-0091BD?logo=xilinx&logoColor=white)](hardware/zynq7000/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

[![Launch Application](https://img.shields.io/badge/Launch%20Application-bharath--ai--soc.onrender.com-00C7B7?style=for-the-badge&logo=render&logoColor=white)](https://bharath-ai-soc.onrender.com)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in%20Codespaces-Cloud%20Dev-blue?style=for-the-badge&logo=github)](https://codespaces.new/keer9999/BHARATH-AI-SOC?quickstart=1)

</div>

---

## Table of Contents

- [System Overview](#system-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Slicing Aided Hyper Inference (SAHI) Engine](#slicing-aided-hyper-inference-sahi-engine)
- [Multi-Model Specialist Ensemble](#multi-model-specialist-ensemble)
- [Hardware Actuation and Pin Mapping](#hardware-actuation-and-pin-mapping)
- [Edge Dashboard and REST API](#edge-dashboard-and-rest-api)
- [Repository Structure](#repository-structure)
- [Installation and Quick Start](#installation-and-quick-start)
- [Verification and Test Suite](#verification-and-test-suite)
- [Technology Stack](#technology-stack)
- [Competition Alignment](#competition-alignment)
- [License](#license)

---

## System Overview

EcoSort AI is an industrial-grade, fully on-device multimodal AI platform designed for real-time waste material perception, small-object detection, and automated circular disposal guidance. The system executes both computer vision and generative language reasoning locally on edge System-on-Chip (SoC) architectures (including Xilinx Zynq-7000 dual-core ARM Cortex-A9, Raspberry Pi 4B/5, and Jetson platforms), eliminating cloud dependencies, external latency, and data transmission vulnerabilities.

Developed for Problem Statement 2 of the Bharat AI-SoC Challenge 2026-27, the platform integrates:
- High-recall object detection powered by a 10-model specialist neural ensemble.
- A native Slicing Aided Hyper Inference (SAHI) engine using Weighted Box Fusion (WBF) for dense and small-scale item detection.
- Context-aware local language reasoning with Meta Llama 3.2 1B and a deterministic 2,300-rule expert knowledge base (WasteAdvisor).
- Physical bin routing via memory-mapped AXI GPIO actuation on the ZedBoard development platform.

### Multimodal Subsystem Matrix

| Subsystem | Underlying Technology | Primary Functional Output |
|---|---|---|
| Vision Perception | Ultralytics YOLOv8s / YOLO11s | Real-time object localization, bounding box regression, and category classification |
| Sliced Inference (SAHI) | Adaptive Grid Tiling + Pure-NumPy WBF | Small-object resolution enhancement without re-training base neural backbones |
| Language Reasoning | Meta Llama 3.2 1B GGUF / ExecuTorch | On-device contextual disposal guidance, DIY upcycling recipes, and resin analysis |
| Deterministic Rules | WasteAdvisor (2,300 verified rules) | Sub-millisecond polymer identification (RIC #1-#7), decomposition rates, and hazard alerts |
| Hardware Actuation | AXI GPIO / BRAM on Zynq-7000 | Low-latency physical bin diversion via onboard LEDs (LD1-LD7) and robotic relay triggers |

---

## Pipeline Architecture

```
+-------------------------------------------------------------------------------+
|                             EcoSort AI Architecture                           |
|                                                                               |
|  Input Video / High-Resolution Sensor Capture                                 |
|         |                                                                     |
|         v                                                                     |
|  +-------------------------------------------------------------------------+  |
|  | Slicing Aided Hyper Inference (SAHI) Layer                              |  |
|  |                                                                         |  |
|  |   Full Frame Pass (640px) ------+                                       |  |
|  |   Multi-scale Pass (320px) -----+---> Overlap Slices (e.g. 416px, 30%)  |  |
|  |                                 |     |   |   |   |                     |  |
|  |                                 |     v   v   v   v                     |  |
|  |                                 |    10-Model Specialist Ensemble       |  |
|  |                                 |     |   |   |   |                     |  |
|  |                                 +---> Coordinate Re-projection          |  |
|  |                                       |                                 |  |
|  |                                       v                                 |  |
|  |                           Weighted Box Fusion (WBF)                     |  |
|  +-----------------------------------------+-------------------------------+  |
|                                            |                                  |
|                                            v                                  |
|  +-----------------------------------------+-------------------------------+  |
|  | Classification & Routing Core                                           |  |
|  |                                                                         |  |
|  |   Plastic      -> Blue Bin        Textiles/Dry Waste -> Green Bin       |  |
|  |   Glass        -> Emerald Bin     Hazardous/E-Waste  -> Red Bin         |  |
|  |   Metal        -> Grey Bin        Paper/Cardboard    -> White/Amber Bin |  |
|  |   Organic/Food -> Brown Bin                                             |  |
|  +-----------------------------------------+-------------------------------+  |
|                   |                                    |                      |
|                   v                                    v                      |
|  +---------------------------------+  +------------------------------------+  |
|  | Hardware Actuation (AXI GPIO)   |  | Reasoning & Advisory Modality      |  |
|  |                                 |  |                                    |  |
|  | Zynq-7000 XC7Z020 (ZedBoard)    |  | Llama 3.2 1B (GGUF / ExecuTorch)   |  |
|  | Physical LED Routing (LD1-LD7)  |  | + WasteAdvisor (2,300 Rules)       |  |
|  +---------------------------------+  +------------------------------------+  |
|                                                        |                      |
|                                                        v                      |
|                                       +------------------------------------+  |
|                                       | Edge Web Dashboard (Flask + REST)  |  |
|                                       | Telemetry, HUD, Analytics, Chatbot |  |
|                                       +------------------------------------+  |
+-------------------------------------------------------------------------------+
```

---

## Slicing Aided Hyper Inference (SAHI) Engine

Industrial conveyor waste segregation frequently encounters tiny items (bottle caps, plastic pull tabs, medication blisters, fragmented electronics) that standard fixed-input convolutional and transformer detectors miss due to downsampling.

EcoSort AI implements an integrated, framework-independent SAHI engine in `src/sahi_detector.py`:

1. **Adaptive Window Tiling:** The input frame is decomposed into overlapping square patches based on resolution and hardware profile parameters.
2. **Boundary Padding:** Edge tiles are padded with letterbox margins to preserve geometric aspect ratios for anchor matching.
3. **Parallel / Sequential Patch Inference:** The primary detector and specialist ensemble evaluate each tile independently.
4. **Coordinate De-normalization:** Detections from tile-space are mathematically mapped back to absolute frame coordinates.
5. **Weighted Box Fusion (WBF):** Rather than standard Non-Maximum Suppression (NMS) which discards overlapping candidate boxes, WBF computes confidence-weighted coordinates across overlapping patches:
   $$\text{FusedBox} = \frac{\sum_{i=1}^{N} c_i \cdot \text{Box}_i}{\sum_{i=1}^{N} c_i}$$
   This results in higher localization precision and superior recall in dense waste piles.

### Hardware-Tuned SAHI Profiles

| Profile Name | Tile Size | Overlap Ratio | Max Tiles | Fusion Strategy | Frame Throttle | Target Hardware |
|---|---|---|---|---|---|---|
| `rpi4` | 320 px | 25% | 4 | Hard NMS | 500 ms (2 FPS) | Raspberry Pi 4B (ARM Cortex-A72, 4GB) |
| `jetson` | 416 px | 25% | 9 | WBF | None | NVIDIA Jetson Nano / Orin Nano (FP16) |
| `gpu` | 640 px | 30% | 16 | WBF + Multiscale | None | Discrete Workstation GPU / Server |
| `default` | 416 px | 30% | 9 | WBF + Multiscale | None | Standard CPU / Core i5 / Cloud Container |

SAHI execution can be enabled at launch (`--sahi --sahi-profile rpi4`) or toggled dynamically via the web interface or REST API without restarting the application.

---

## Multi-Model Specialist Ensemble

To balance latency constraints against granular material classification, EcoSort AI features a 10-model specialist architecture:

| Role Identifier | Model Source File | Target Categories | Functional Specialization |
|---|---|---|---|
| `primary_ecosort` | `ecosort_waste_yolo11s.pt` | 66 Classes | Primary detector covering granular recyclables and packaging |
| `primary_best` | `best.pt` | 5 Classes | Fast-path binary/quinary sorting for ARM Cortex-A9 core |
| `ewaste_a` | `ewaste_a.pt` | 12 Classes | Compact electronic waste (keyboards, cells, circuit boards) |
| `ewaste_b` | `ewaste_b.pt` | 77 Classes | Comprehensive consumer and industrial electronic waste |
| `plastic_bag` | `plastic_bag.pt` | 1 Class | Single-use carrier bags, stretch films, polythene packaging |
| `general_waste` | `general_waste.pt` | 5 Classes | General municipal solid waste baseline verification |
| `plastic_containers`| `plastic_containers.pt` | 3 Classes | Beverage bottles, food tubs, disposable cups |
| `resin_codes` | `resin_codes.pt` | 6 Classes | Resin Identification Code recognition (HDPE, LDPE, PETE, PP, PS, PVC) |
| `food_organic` | `food_organic.pt` | 39 Classes | Wet organic scraps, compostables, and discarded textiles |
| `bottle` | `bottle_specialist.pt` | 1 Class | High-precision single-use beverage container localization |

### Canonical Material Taxonomy

All detections are normalized into canonical segregation classes:
- `recyclable_plastic`: PET (#1), HDPE (#2), PVC (#3), LDPE (#4), PP (#5), PS (#6), and polymers.
- `recyclable_glass`: Clear flint, green, and amber container glass.
- `recyclable_metal`: Ferrous cans, non-ferrous aluminum cans, tinplate, foil.
- `recyclable_paper_cardboard`: Corrugated boxes, cartonboard, office paper, newsprint.
- `organic_food`: Kitchen waste, vegetable scraps, plant trimmings, compostables.
- `dry_waste`: Textiles, footwear, fabric scraps, wooden and dry utility waste.
- `hazardous_ewaste`: Batteries, printed circuit boards, electronic appliances, fluorescent lamps.

---

## Hardware Actuation and Pin Mapping

On the Xilinx Zynq-7000 (XC7Z020-CLG484-1) ZedBoard, perception decisions are routed directly to physical AXI-mapped GPIO registers:

| Signal Name | Physical LED | Pin Location | Color Representation | Category Route |
|---|---|---|---|---|
| `PIN_LED1` | LD1 | T22 | Blue Indicator | Recyclable Plastic |
| `PIN_LED2` | LD2 | T21 | Green Indicator | Recyclable Glass |
| `PIN_LED3` | LD3 | U22 | Grey / Amber Indicator | Recyclable Metal |
| `PIN_LED4` | LD4 | U21 | White Indicator | Paper and Cardboard |
| `PIN_LED5` | LD5 | V22 | Brown Indicator | Organic / Wet Waste |
| `PIN_LED6` | LD6 | W22 | Violet Indicator | Electronic Waste (E-Waste) |
| `PIN_LED7` | LD7 | U19 | Red Indicator | Hazardous / Chemical Waste |

The bare-metal implementation in `hardware/zynq7000/bram/` utilizes direct `/dev/mem` pointer operations to ensure deterministic, sub-microsecond GPIO assertion.

---

## Edge Dashboard and REST API

The platform includes an embedded web dashboard and RESTful API implemented in Flask:

### REST API Endpoints

| Method | Endpoint | Description | Sample Request / Parameters |
|---|---|---|---|
| `GET` | `/health` | Service health status | None |
| `GET` | `/api/status` | System health, camera, and runtime metrics | None |
| `POST` | `/api/detect` | Multi-model waste detection on image input | Multipart `image` file or JSON base64 |
| `GET` | `/api/webcam/stream` | Multi-client shared MJPEG video stream with HUD | None (Stream) |
| `POST` | `/api/webcam/capture`| Capture single frame and execute full inference | None |
| `POST` | `/api/chat` | Multimodal conversational disposal advisory | JSON: `{"message": "How do I recycle PET?"}` |
| `GET` | `/api/model/info` | Loaded weights, taxonomy, and SAHI statistics | None |
| `GET` | `/api/sahi/status` | SAHI operational status and tiling configuration | None |
| `POST` | `/api/sahi/toggle` | Enable/disable SAHI or adjust hardware profile | JSON: `{"enabled": true, "profile": "rpi4"}` |
| `GET` | `/api/stats` | Aggregated waste volume and class distribution | Query: `?period=today` |
| `GET` | `/api/records` | Tabular detection event history | Query: `?limit=50` |

---

## Repository Structure

```
BHARATH-AI-SOC/
├── src/                                  # Application Source Tree
│   ├── main.py                           # CLI execution entry point
│   ├── web_app.py                        # Web dashboard and REST API backend
│   ├── waste_detector.py                 # Multi-model YOLO detector and HUD compositor
│   ├── sahi_detector.py                  # SAHI engine, grid slicing, and WBF implementation
│   ├── waste_advisor.py                  # 2,300-rule expert advisory knowledge engine
│   ├── llama_engine.py                   # On-device Llama 3.2 GGUF inference interface
│   ├── executorch_llama_engine.py        # ExecuTorch mobile runtime loader
│   ├── config.py                         # Central system configuration and class mappings
│   ├── database.py                       # SQLite database interface for telemetry
│   ├── report_generator.py               # Automated PDF and CSV report generator
│   ├── webcam_capture.py                 # Threaded OpenCV camera acquisition hub
│   ├── train_waste_model.py              # Transfer learning and training pipeline
│   ├── web/
│   │   ├── templates/
│   │   │   ├── index.html                # Responsive web interface
│   │   │   └── 404.html                  # Error view
│   │   └── static/
│   │       ├── css/style.css             # Interface styling and SAHI badge styles
│   │       └── js/
│   │           ├── app.js                # Core frontend interaction logic
│   │           └── sahi.js               # SAHI controller, status polling, and toggle logic
│   └── yolo/
│       └── weights/                      # Local storage for neural model checkpoints (.pt)
│
├── hardware/
│   ├── zynq7000/                         # Primary Competition Embedded Target
│   │   └── bram/
│   │       ├── main_zynq.py              # Dual ARM Cortex-A9 execution and GPIO driver
│   │       └── config_zynq.py            # Memory mapping and peripheral definitions
│   └── hailo_npu/                        # Optional Hailo-8 NPU acceleration drivers
│
├── models/
│   ├── unified/                          # Consolidated single-file deployment artifacts
│   │   ├── c2s_waste_ensemble.pt         # Unified 10-model PyTorch package
│   │   ├── build_report.json             # Build metadata and model manifest
│   │   └── README.md                     # Model serialization documentation
│   ├── best.pt                           # 5-class primary classifier
│   ├── ecosort_waste_yolo11s.pt          # 66-class primary detector
│   ├── weights.pt                        # Beverage container specialist
│   └── yolov8n.pt                        # Lightweight general fallback
│
├── tests/                                # Automated Test and Validation Suite
│   ├── conftest.py                       # Shared test fixtures and mocking setup
│   ├── test_sahi_integration.py          # SAHI geometry, WBF fusion, and API test cases
│   ├── test_v2_taxonomy.py               # 7-class taxonomy verification
│   ├── test_web_hardening.py             # Web server concurrency, payload, and thread-safety tests
│   ├── test_waste_advisor.py             # Knowledge engine rule verification
│   ├── test_material_mapping.py          # Class normalization checks
│   ├── test_executorch_llama.py          # Llama engine integration checks
│   └── test_config_paths.py              # Configuration integrity tests
│
├── scripts/
│   ├── setup.sh                          # Automated environment bootstrap script
│   ├── download_models.sh                # Automated model weight synchronization
│   ├── build_unified_model.py            # Script to consolidate specialist models into .pt
│   └── run_tests.sh                      # Shell test runner
│
├── docs/                                 # Architectural and Deployment Documentation
│   ├── ARCHITECTURE.md                   # System technical design document
│   ├── API_REFERENCE.md                  # Comprehensive REST endpoint specifications
│   ├── HARDWARE_SETUP.md                 # ZedBoard and sensor wiring guide
│   ├── TRAINING.md                       # Neural network training logs and hyperparameters
│   ├── DEPLOYMENT.md                     # Production deployment documentation
│   └── PROJECT_REPORT.md                 # Comprehensive engineering report
│
├── .devcontainer/devcontainer.json       # GitHub Codespaces container definition
├── Dockerfile                            # Production container image specification
└── render.yaml                           # Cloud deployment configuration
```

---

## Installation and Quick Start

### Method 1: Local Development Setup (Linux / macOS / WSL)

```bash
# 1. Clone the repository
git clone https://github.com/keer9999/BHARATH-AI-SOC.git
cd BHARATH-AI-SOC

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r src/requirements.txt

# 4. Synchronize pre-trained model weights
bash scripts/download_models.sh

# 5. Launch the application (Standard mode)
python src/web_app.py --port 5000

# 5b. Launch the application with SAHI enabled
python src/web_app.py --port 5000 --sahi --sahi-profile default
```

Open `http://localhost:5000` (or `http://localhost:5050` if port 5000 is occupied) in any modern web browser.

### Method 2: Command Line Interface (CLI)

```bash
# Single image inference with full disposal advice
python src/main.py --image test_image.jpg

# Headless continuous camera inspection
python src/main.py --headless

# Comprehensive system self-test
python src/main.py --self-test
```

### Method 3: Embedded Zynq-7000 Deployment (ZedBoard)

```bash
# Verify AXI GPIO register mapping and test LEDs LD1-LD7
python hardware/zynq7000/bram/main_zynq.py --test

# Run continuous hardware-accelerated classification loop
python hardware/zynq7000/bram/main_zynq.py
```

---

## Verification and Test Suite

The project includes an automated test suite covering classification taxonomy, edge web hardening, rule verification, and SAHI sliced inference.

To execute the test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

All 58 test cases execute without requiring a physical camera, GPU, or active network connection:
- `test_sahi_integration.py`: Validates tile coordinate computation, letterbox square padding, pure-NumPy Weighted Box Fusion, hard NMS, hardware profiles, and REST API endpoints.
- `test_v2_taxonomy.py`: Validates zero-regression legacy classification and correct behavior of the extended 7-class material taxonomy.
- `test_web_hardening.py`: Validates maximum upload payload guards (25 MB), bad-input HTTP 400 responses, single-camera hub concurrency, and single-inference locking.
- `test_waste_advisor.py`: Validates circular economy advice, resin identification, and decomposition queries.
- `test_material_mapping.py`: Validates raw string token normalization across edge cases.
- `test_executorch_llama.py`: Validates prompt assembly and fallback reasoning paths.

---

## Technology Stack

| Domain | Component | Technical Role |
|---|---|---|
| Perception | Ultralytics YOLOv8 / YOLO11 | Real-time object localization and classification |
| Slicing Inference | SAHI Engine | Tiled inference with pure-NumPy Weighted Box Fusion |
| Language AI | Meta Llama 3.2 1B (GGUF) | On-device conversational reasoning |
| Rule Engine | WasteAdvisor | Deterministic circular economy and resin code analysis |
| Embedded Target | Xilinx Zynq-7000 (ZedBoard) | Dual-core ARM Cortex-A9 with AXI GPIO actuation |
| Web Platform | Flask 3.x, HTML5, CSS3, JavaScript | Dashboard, telemetry visualization, and REST endpoints |
| Containerization | Docker, GitHub Codespaces | Reproducible evaluation and deployment |

---

## Competition Alignment

| Parameter | Specification | Implementation in EcoSort AI |
|---|---|---|
| Competition | Bharat AI-SoC Challenge 2026-27 | Fully compliant with competition rules and deadlines |
| Problem Statement | Problem Statement 2 (PS2) | AI-Powered Edge Waste Classification and Disposal Assistant |
| Organizers | Arm, MeitY C2S, IIT Delhi | Academic and industrial rigor maintained throughout |
| Target SoC | Arm-based System-on-Chip | Native dual-core ARM Cortex-A9 deployment (Zynq-7000) |
| Multimodality | Computer Vision + Natural Language | Vision perception (YOLO+SAHI) combined with on-device LLM reasoning |
| Cloud Independence | Zero Cloud Offloading | Entire vision and language pipeline runs locally on device |

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete licensing terms.
