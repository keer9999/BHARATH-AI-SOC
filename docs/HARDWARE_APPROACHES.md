# Hardware Architecture Comparison & Selection Guide
## Bharat AI-SoC Challenge 2026–27 | Problem Statement 2 (PS-2)
**AI-Powered Edge Waste Classification and Disposal Assistant**  
*Organized by Arm, MeitY C2S Programme, and IIT Delhi*

---

## 1. Executive Summary

Problem Statement 2 (PS-2) mandates an **on-device, cloud-independent AI system** deployed on **Arm-based System-on-Chip (SoC)** architectures for real-time waste classification, sensor fusion, and automated disposal assistance.

To address both **academic/hardware rigor** (demanded by FPGA/SoC evaluation rubrics) and **commercial/field deployment viability** (cost, throughput, and turnkey reproduction), EcoSort AI provides two fully implemented hardware deployment architectures:

1. **Approach 1 (Heterogeneous Co-Design):** Xilinx Zynq-7000 (ZedBoard / ZC702) + Raspberry Pi 4B/5
2. **Approach 2 (Monolithic Edge SoC):** Standalone Raspberry Pi 4B / 5

Both approaches utilize the **unified 66-class YOLO11s ONNX model** (`ecosort_waste_yolo11s.onnx`), the **Llama 3.2 1B on-device reasoning engine** (GGUF / ExecuTorch), and the **Raspberry Pi Pico W sensor node**.

---

## 2. Approach 1: Heterogeneous Co-Design (Zynq-7000 ZedBoard + Raspberry Pi)

### 2.1 Architectural Overview

Approach 1 splits the computational burden across two specialized domains:
- **Compute Plane (Raspberry Pi 4B/5):** High-throughput neural network perception (YOLO11s ONNX) and language model reasoning (Llama 3.2 1B).
- **Actuation Plane (Xilinx Zynq-7000 ZedBoard):** Hard real-time hardware actuation, physical bin diversion, and failsafe monitoring implemented across the Processing System (PS: Dual ARM Cortex-A9 @ 667 MHz) and Programmable Logic (PL: Artix-7 FPGA fabric).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              APPROACH 1: HETEROGENEOUS PS + PL HIERARCHICAL SOC ARCHITECTURE     │
│                                                                                 │
│   ┌────────────────────────┐                                                    │
│   │  Raspberry Pi Pico W   │                                                    │
│   │  (MicroPython Sensors) │                                                    │
│   │  HC-SR04 · DHT11       │                                                    │
│   │  MQ-135 · Moisture     │                                                    │
│   └───────────┬────────────┘                                                    │
│               │ USB-CDC JSON @ 115200 baud                                      │
│               ▼                                                                 │
│   ┌──────────────────────────────────────────────────┐                          │
│   │  Raspberry Pi 4B / 5 Compute Node (ARM Cortex)   │                          │
│   │  ──────────────────────────────────────────────  │                          │
│   │  • Unified 66-Class YOLO11s ONNX (~160–300 ms)   │                          │
│   │  • SAHI Slicing Aided Hyper Inference Engine     │                          │
│   │  • Llama 3.2 1B Instruct (GGUF / ExecuTorch)     │                          │
│   │  • Flask Edge Web Dashboard (Port 5001)          │                          │
│   └───────────────────────────┬──────────────────────┘                          │
│                               │ High-Speed UART / SPI Inter-Board Link          │
│                               ▼                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Xilinx Zynq-7000 ZedBoard (Dual ARM Cortex-A9 + Artix-7 FPGA Fabric)  │   │
│   │                                                                         │   │
│   │   [ Processing System (PS) - ARM Cortex-A9 ]                           │   │
│   │   • PetaLinux 2024.1 Embedded OS                                        │   │
│   │   • `main_zynq.py` Real-Time Actuation Controller                       │   │
│   │   • Failsafe Watchdog & Telemetry Logging                               │   │
│   │               │                                                         │   │
│   │               │ AXI4-Lite Memory-Mapped Bus                             │   │
│   │               ▼                                                         │   │
│   │   [ Programmable Logic (PL) - FPGA Fabric ]                             │   │
│   │   • AXI GPIO Controller (Base: 0x41200000)                              │   │
│   │   • Direct BRAM Routing Registers                                       │   │
│   │   • Deterministic Diverter Pulse Generator (< 1 µs Jitter)              │   │
│   │               │                                                         │   │
│   │               ▼                                                         │   │
│   │   [ Physical Outputs ]                                                  │   │
│   │   • LD1–LD7 Physical Status LEDs                                        │   │
│   │   • Optical Isolation Relays → Pneumatic Flaps / Servos                 │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Zynq-7000 Memory Map & Hardware Pinout

The Zynq Programmable Logic implements an AXI GPIO core configured via Vivado (`hardware/zynq7000/vivado/zedboard_block_design.tcl`):

| Physical Output | Waste Category | Bin Color | AXI Bit Mask | Physical Pin (ZedBoard) |
|---|---|---|---|---|
| **LD1** | Plastic | Blue | `0x00000001` | Pin T22 |
| **LD2** | Glass | Green | `0x00000002` | Pin T21 |
| **LD3** | Metal | Grey | `0x00000004` | Pin U22 |
| **LD4** | Paper / Cardboard | Yellow | `0x00000008` | Pin U21 |
| **LD5** | Organic / Food Waste | Brown | `0x00000010` | Pin V22 |
| **LD6** | E-Waste | Purple | `0x00000020` | Pin W22 |
| **LD7** | Hazardous | Red | `0x00000040` | Pin U19 |

### 2.3 Key Strengths of Approach 1

1. **Direct Alignment with Arm AI-SoC Mandate:** Implements heterogeneous PS+PL co-design on a dual ARM Cortex-A9 SoC platform, fulfilling the core hardware objective of the challenge.
2. **Sub-Microsecond Actuation Determinism:** Diverter switching and relay timing operate via hardware AXI registers without Linux scheduler latency or garbage collection interruptions.
3. **Safety & Fault Isolation:** If the high-level Python/web stack crashes or reboots, the Zynq PL maintains safety interlocks, bin capacity alarms, and hardware watchdog states.

---

## 3. Approach 2: Monolithic Edge SoC (Standalone Raspberry Pi 4B / 5)

### 3.1 Architectural Overview

Approach 2 consolidates the entire perception, reasoning, data fusion, and user interface into a single **Raspberry Pi 4B (4/8 GB)** or **Raspberry Pi 5 (8 GB)**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│               APPROACH 2: MONOLITHIC STANDALONE RASPBERRY PI 4B/5               │
│                                                                                 │
│   ┌────────────────────────┐                                                    │
│   │  Raspberry Pi Pico W   │                                                    │
│   │  (MicroPython Sensors) │                                                    │
│   │  HC-SR04 · DHT11       │                                                    │
│   │  MQ-135 · Moisture     │                                                    │
│   └───────────┬────────────┘                                                    │
│               │ USB Serial (/dev/ttyACM0)                                       │
│               ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Raspberry Pi 4B / 5 (Quad-Core 64-bit ARM Cortex-A72 / A76)            │   │
│   │                                                                         │   │
│   │  [ Ingestion & Fusion Layer ]                                           │   │
│   │  • USB Webcam Capture (V4L2 720p @ 30 FPS)                              │   │
│   │  • `data_fusion.py`: Correlates visual detections with moisture/gas     │   │
│   │                                                                         │   │
│   │  [ AI Perception Engine ]                                               │   │
│   │  • `ecosort_waste_yolo11s.onnx` via ONNX Runtime CPU (~160 ms on Pi 5)  │   │
│   │  • SAHI Slicing Aided Hyper Inference + Weighted Box Fusion (WBF)       │   │
│   │                                                                         │   │
│   │  [ Contextual Reasoning Engine ]                                        │   │
│   │  • Llama 3.2 1B Instruct GGUF (`llama-cpp-python` ARM NEON)             │   │
│   │  • Optional: ExecuTorch .pte with XNNPACK runtime (2× faster on Pi 5)   │   │
│   │  • `waste_advisor.py`: 2,300-rule sub-5ms deterministic expert fallback │   │
│   │                                                                         │   │
│   │  [ Actuation & Interface Layer ]                                        │   │
│   │  • Software GPIO (`RPi.GPIO` / `libgpiod`) → Relay Module               │   │
│   │  • Flask Edge Web Dashboard (Port 5001) → Accessible over local Wi-Fi   │   │
│   │  • SQLite On-Device Database (`waste_data.db`)                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Strengths of Approach 2

1. **Ultra-Low Bill of Materials (BOM):** Single-board computer (~$60–$80), making mass municipal deployment economically feasible.
2. **Turnkey Reproducibility:** Anyone (including competition evaluators) can clone the repository, run `bash raspberry_pi/keerthans_pi/scripts/setup_pi.sh`, and have the entire system running in under 5 minutes.
3. **High Power Efficiency:** Total system power consumption under full inference load is **~6.5W on Pi 4B** and **~11W on Pi 5**, making it suitable for solar/battery-powered mobile collection vehicles.
4. **Arm Cortex-A76 Vector Extensions:** Raspberry Pi 5 incorporates modern ARMv8.2-A vector extensions (Dot Product & FP16 arithmetic), accelerating quantized INT8/INT4 neural network inference via XNNPACK.

---

## 4. Side-by-Side Technical Comparison Matrix

| Evaluation Dimension | Approach 1: Zynq-7000 + Raspberry Pi | Approach 2: Standalone Raspberry Pi 4B/5 |
|---|---|---|
| **SoC Processor Core** | Dual ARM Cortex-A9 (Zynq) + Quad ARM Cortex-A72/A76 (Pi) | Quad ARM Cortex-A72 (Pi 4B) / Cortex-A76 (Pi 5) |
| **Hardware Architecture** | Heterogeneous PS + PL (FPGA co-design) | Homogeneous Multicore Edge SoC |
| **YOLO11s ONNX Latency** | ~160 ms (Pi 5) / ~320 ms (Pi 4B) | ~160 ms (Pi 5) / ~320 ms (Pi 4B) |
| **Actuation Latency** | **< 1 µs** (AXI4-Lite register mapping) | ~2–5 ms (Linux sysfs / libgpiod) |
| **Actuation Determinism** | **Hard Real-Time** (zero jitter, hardware clock) | Soft Real-Time (kernel scheduler dependent) |
| **Language Model Runtime** | Llama 3.2 1B (Pi) + WasteAdvisor (Zynq PS) | Llama 3.2 1B (GGUF / ExecuTorch) + WasteAdvisor |
| **Total System RAM** | 1 GB DDR3 (ZedBoard) + 4–8 GB LPDDR4 (Pi) | 4 GB or 8 GB LPDDR4X |
| **System Power Consumption** | ~14W – 18W (ZedBoard + Pi + Sensors) | **~5W – 11W** (Pi + Sensors) |
| **Bill of Materials (BOM)** | ~$550 (ZedBoard: ~$475 + Pi: ~$75) | **~$75 – $95** (Raspberry Pi 4B/5 + Pico W) |
| **Toolchain Dependencies** | Xilinx Vivado 2024.1 + PetaLinux 2024.1 + Python | Standard Linux (Raspberry Pi OS 64-bit) + Python |
| **Setup & Evaluation Time** | ~45 minutes (SD card flash + serial console) | **~5 minutes** (One-shot setup script) |
| **Competition Alignment** | **Highest Academic & SoC Architecture Rigor** | **Highest Practical Viability & Reproducibility** |

---

## 5. Official Submission Strategy & Justification

### Why presenting both approaches is optimal:

1. **Satisfies the Dual Evaluation Criteria:**
   - Evaluators focusing on **SoC co-design, AXI bus architectures, and hardware-software partitioning** can inspect [`hardware/zynq7000/`](hardware/zynq7000/) for full Vivado block designs, pin constraints, and PetaLinux recipes.
   - Evaluators focusing on **live demonstration, real-world accuracy, throughput, and ease of deployment** can evaluate [`raspberry_pi/keerthans_pi/`](raspberry_pi/keerthans_pi/) on any standard Raspberry Pi with zero friction.

2. **Unified Software Foundation:**
   - Both approaches share identical model weights (`ecosort_waste_yolo11s.onnx`), identical sensor telemetry schemas (Raspberry Pi Pico W JSON), and identical expert disposal intelligence (`src/waste_advisor.py` + Llama 3.2 1B).
   - Upgrading from Approach 2 to Approach 1 requires zero modifications to the computer vision model or training pipeline.

3. **Production Recommendation:**
   - **For stationary municipal sorting facilities (MRFs):** **Approach 1** is recommended for high-speed conveyor belts requiring microsecond pneumatic sorting triggers.
   - **For mobile collection vehicles (smart trucks) and commercial bins:** **Approach 2** is recommended for low power, low cost, and compact physical footprint.
