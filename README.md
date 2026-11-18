<div align="center">

<img src="https://img.shields.io/badge/Bharat%20AI--SoC-2026--27-FF6B00?style=for-the-badge&logo=arm&logoColor=white" alt="Bharat AI-SoC"/>

# EcoSort AI
### On-Device Multimodal Waste Segregation & Circular Recycling Platform

*Problem Statement 2 — AI-Powered Edge Waste Classification and Disposal Assistant*  
*Organised by Arm x MeitY C2S x IIT Delhi*

<br/>

[![CI](https://github.com/keer9999/BHARATH-AI-SOC/actions/workflows/ci.yml/badge.svg)](https://github.com/keer9999/BHARATH-AI-SOC/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Vision](https://img.shields.io/badge/Vision-YOLOv8%20%7C%20YOLO11-00FFFF?logo=ultralytics&logoColor=black)](https://docs.ultralytics.com/)
[![LLM](https://img.shields.io/badge/LLM-Llama%203.2%201B-6C2FBF?logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![Hardware](https://img.shields.io/badge/SoC-Zynq--7000%20%7C%20ARM%20Cortex--A9-0091BD?logo=xilinx&logoColor=white)](hardware/zynq7000/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

[![Launch Application](https://img.shields.io/badge/Launch%20Application-bharath--ai--soc.onrender.com-00C7B7?style=for-the-badge&logo=render&logoColor=white)](https://bharath-ai-soc.onrender.com)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in%20Codespaces-Cloud%20Dev-blue?style=for-the-badge&logo=github)](https://codespaces.new/keer9999/BHARATH-AI-SOC?quickstart=1)

<br/>

> **[Launch Live Application (https://bharath-ai-soc.onrender.com)](https://bharath-ai-soc.onrender.com)**  
> Deployed on cloud infrastructure with zero local installation required.

<br/>

</div>

---

## System Overview

**EcoSort AI** is an industrial-grade, fully on-device multimodal AI platform engineered for real-time waste identification and circular disposal guidance. It executes the perception and reasoning pipeline locally on edge hardware (Xilinx Zynq-7000 ARM Cortex-A9), eliminating cloud latency and privacy risks.

Developed for **Problem Statement 2** of the **Bharat AI-SoC Challenge 2026-27**, the platform demonstrates edge multimodality combining high-speed computer vision with local generative language reasoning.

### Multimodal Architecture Matrix

| Subsystem | Core Engine | Functional Output |
|---|---|---|
| **Vision Perception** | YOLOv8 / YOLO11s (66+ classes) | Real-time object detection, spatial bounding, and class confidence |
| **Language Reasoning** | Llama 3.2 1B & WasteAdvisor Engine | Material decomposition timelines, circular upcycling, and hazard advisories |
| **Hardware Actuation** | Zynq-7000 ZedBoard GPIO / BRAM | Low-latency physical bin routing via on-board LEDs (LD1-LD7) |

---

## Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EcoSort AI Pipeline                             │
│                                                                        │
│   Video Stream / High-Res Frame                                        │
│          │                                                             │
│          ▼                                                             │
│   ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐   │
│   │  Modality 1  │     │   ROUTING CORE   │     │  Modality 2      │   │
│   │  PERCEPTION  │────▶│  PLASTIC -> BLUE │────▶│  REASONING       │   │
│   │              │     │  GLASS   -> GREEN│     │                  │   │
│   │  YOLOv8 /    │     │  METAL   -> GREY │     │  Llama 3.2 1B    │   │
│   │  YOLO11s     │     │  PAPER   -> WHITE│     │  WasteAdvisor KB │   │
│   │  (66 Class)  │     │  ORGANIC -> BROWN│     │  (2,300 Rules)   │   │
│   └──────────────┘     └──────────────────┘     └──────────────────┘   │
│                                │                         │             │
│                                ▼                         ▼             │
│                     ┌────────────────────┐    ┌─────────────────────┐  │
│                     │  Zynq-7000 SoC     │    │  Edge Dashboard     │  │
│                     │  ZedBoard LEDs     │    │  Live Video HUD +   │  │
│                     │  LD1-LD7 Hardware  │    │  Analytics + Chat   │  │
│                     └────────────────────┘    └─────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

**Edge-First Design Principle:** Inference runs on the dual ARM Cortex-A9 cores without internet dependency, external APIs, or cloud offloading.

---

## Ensemble Model Architecture

EcoSort AI employs a hierarchical ensemble of four specialized neural models to balance latency and granularity:

| Model Artifact | Architecture | Parameters & Classes | Operational Role |
|---|---|---|---|
| `best.pt` | YOLOv8s (22.5 MB) | 5 Classes (Plastic, Glass, Metal, Paper, Organic) | **Primary Classifier** — Fast-path bin routing on ZedBoard |
| `ecosort_waste_yolo11s.pt` | YOLO11s (19.2 MB) | 66 Classes (Aerosol, Battery, Can, Carton, etc.) | **Fine-Grained Engine** — Supplies semantic context to Llama |
| `weights.pt` | YOLOv8s (20.3 MB) | 1 Class (Container / Bottle Specialist) | **Container Specialist** — Maximizes detection recall for beverage packaging |
| `yolov8n.pt` | YOLOv8n (6.5 MB) | 80 COCO Classes with waste remap | **Fallback Backbone** — Ultra-low-power execution on compute-limited cores |

---

## ZedBoard Hardware LED Routing

On the physical ZedBoard (XC7Z020), AXI-mapped GPIO LEDs provide instant actuation for automated bin deflection or operator guidance:

| Hardware LED | Target Stream | Container Indicator | Exemplar Materials |
|---|---|---|---|
| **LD1** | Plastic Recyclable | Blue Bin | PET bottles, HDPE jugs, PP containers |
| **LD2** | Glass Recyclable | Green Bin | Beverage bottles, food jars, cullet |
| **LD3** | Metal Recyclable | Grey Bin | Aluminium beverage cans, food tins, scrap foil |
| **LD4** | Paper & Fibers | White Bin | Corrugated cartons, newsprint, office paper |
| **LD5** | Organic / Compost | Brown Bin | Food scraps, fruit peels, compostable solids |
| **LD6** | Electronic Waste | Red Bin | Lithium cells, circuit boards, peripheral wiring |
| **LD7** | Hazardous Materials | Orange Bin | Pressurized aerosols, solvents, bio-medical items |

---

## Quick Start

### Option A — Live Web Application (Launch in Browser)

Access the live deployed EcoSort AI platform immediately:
**[https://bharath-ai-soc.onrender.com](https://bharath-ai-soc.onrender.com)**

- **Service Status:** [https://bharath-ai-soc.onrender.com/health](https://bharath-ai-soc.onrender.com/health)
- **Features Available:** Full multimodal interface, image upload classification, interactive WasteAdvisor disposal recommendations, and real-time segregation telemetry.

### Option B — Cloud Development (GitHub Codespaces)

1. Click **[Open in GitHub Codespaces](https://codespaces.new/keer9999/BHARATH-AI-SOC?quickstart=1)**.
2. Codespaces automatically provisions the Linux container and dependencies.
3. The platform opens on forwarded port **5000** automatically.

### Option C — Local Workstation Setup (Linux / macOS)

```bash
# 1. Clone repository
git clone https://github.com/keer9999/BHARATH-AI-SOC.git
cd BHARATH-AI-SOC

# 2. Automated environment setup
bash scripts/setup.sh

# 3. Synchronize neural weights
bash scripts/download_models.sh

# 4. Launch web platform
source .venv/bin/activate
python src/web_app.py --host 0.0.0.0 --port 5000
```

Access the local dashboard at **`http://localhost:5000`**.

### Option D — Embedded Zynq-7000 Execution (ZedBoard ARM)

```bash
# Run bare-metal hardware validation with on-board LED feedback
python hardware/zynq7000/bram/main_zynq.py --test

# Execute continuous camera stream segregation on ARM Cortex-A9
python hardware/zynq7000/bram/main_zynq.py
```

---

## Edge Dashboard Capabilities

The integrated Flask web platform (`src/web_app.py`) provides full multimodal control:

| Capability | Technical Implementation | Operational Purpose |
|---|---|---|
| **Real-Time Video Stream** | Low-latency MJPEG stream with dynamic HUD overlays | Continuous conveyor or camera inspection |
| **High-Res Image Analysis** | Multipart file upload with automated inference pipeline | Batch audit of captured waste imagery |
| **Multimodal Advisory Engine** | Llama 3.2 GGUF & 2,300-rule expert knowledge base | Conversational guidance, resin identification, and upcycling |
| **Session Telemetry & Analytics** | SQLite embedded event store with aggregated counts | Material distribution metrics and throughput tracking |
| **High-Speed REST API** | JSON endpoints (`/api/detect`, `/api/chat`, `/api/model/info`) | Machine-to-machine integration for automated sorting rigs |

---

## Repository Structure

```
BHARATH-AI-SOC/
│
├── src/                        # Core Multimodal Engine
│   ├── main.py                 # Primary execution interface (CLI & test runner)
│   ├── web_app.py              # Cloud & edge web dashboard backend
│   ├── waste_detector.py       # YOLO ensemble perception & HUD compositor
│   ├── waste_advisor.py        # 2,300-rule circular advisory knowledge base
│   ├── llama_engine.py         # On-device Llama 3.2 generative reasoning engine
│   ├── config.py               # Centralized configuration & class routing
│   └── web/                    # Dashboard frontend (HTML5 / Modern JS / CSS)
│
├── hardware/
│   ├── zynq7000/               # Primary Competition Target
│   │   └── bram/
│   │       ├── main_zynq.py    # Dual Cortex-A9 entry point with LED actuation
│   │       └── config_zynq.py  # ZedBoard memory & AXI peripheral map
│   └── hailo_npu/              # Optional NPU acceleration modules
│
├── models/                     # Neural Model Registry
│   ├── best.pt                 # 5-class primary bin routing model
│   ├── ecosort_waste_yolo11s.pt# 66-class fine-grained detector
│   ├── weights.pt              # Single-class bottle specialist
│   └── yolov8n.pt              # Lightweight 80-class fallback
│
├── scripts/
│   ├── setup.sh                # End-to-end environment installer
│   ├── download_models.sh      # Neural weight release downloader
│   └── run_tests.sh            # Automated regression test runner
│
├── .devcontainer/              # GitHub Codespaces Cloud Definition
│   └── devcontainer.json
│
├── render.yaml                 # Cloud Deployment Blueprint (Render)
└── Dockerfile                  # Production container definition
```

---

## Technology Stack

| Domain | Framework / Tool | Role in System |
|---|---|---|
| **Computer Vision** | Ultralytics YOLOv8 / YOLO11 | Real-time bounding and multi-class waste perception |
| **Language Intelligence** | Meta Llama 3.2 1B (GGUF) | On-device generative recycling advice |
| **Expert Knowledge** | WasteAdvisor (2,300 rules) | Deterministic polymer identification and circular economy facts |
| **Embedded Platform** | Xilinx Zynq-7000 (ZedBoard) | Dual-Core ARM Cortex-A9 @ 667 MHz with AXI GPIO |
| **Backend & Streaming** | Flask 3.x + OpenCV | Threaded MJPEG feed, REST API, and static UI delivery |
| **Cloud Deployment** | Docker + GitHub Codespaces + Render | Turnkey browser execution for evaluation |

---

## Competition Alignment

| Specification | Project Implementation |
|---|---|
| **Challenge** | Bharat AI-SoC Challenge 2026-27 |
| **Problem Statement** | Problem Statement 2 — AI-Powered Waste Classification & Disposal Assistant |
| **Organizers** | Arm x MeitY C2S Programme x IIT Delhi |
| **Target Architecture** | Arm SoC (Dual-Core Cortex-A9 on Zynq-7000 ZedBoard) |
| **Multimodal Mandate** | Full on-device execution of Vision (YOLO) + Language (Llama 3.2) without cloud APIs |

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for complete terms.

---

<div align="center">

**Built for the Bharat AI-SoC Challenge 2026-27**  
*Arm x MeitY C2S x IIT Delhi*

[![Launch Application](https://img.shields.io/badge/Launch%20Application-bharath--ai--soc.onrender.com-00C7B7?style=for-the-badge&logo=render&logoColor=white)](https://bharath-ai-soc.onrender.com)
[![Open in GitHub Codespaces](https://img.shields.io/badge/Open%20in%20Codespaces-Cloud%20Dev-blue?style=for-the-badge&logo=github)](https://codespaces.new/keer9999/BHARATH-AI-SOC?quickstart=1)

</div>
