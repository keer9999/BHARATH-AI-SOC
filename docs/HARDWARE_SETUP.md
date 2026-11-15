# Hardware Setup Guide

## Overview

This guide covers the full hardware setup for the Bharat AI-SoC PS2 system, from a minimal single-machine setup to the full multi-device edge deployment.

---

## Table of Contents

1. [Minimum Setup (Software Only)](#1-minimum-setup-software-only)
2. [Raspberry Pi 5 Setup](#2-raspberry-pi-5-setup)
3. [Raspberry Pi Pico W Sensor Array](#3-raspberry-pi-pico-w-sensor-array)
4. [Zynq-7000 FPGA Setup](#4-zynq-7000-fpga-setup)
5. [Full System Wiring Diagram](#5-full-system-wiring-diagram)
6. [Power Requirements](#6-power-requirements)

---

## 1. Minimum Setup (Software Only)

Any modern computer with:
- Python 3.11+
- USB webcam (OpenCV-compatible)
- 4 GB RAM (8 GB recommended for Llama 3.2 1B)
- Ubuntu 22.04 LTS or Windows 11 (with WSL2)

```bash
bash scripts/setup.sh
bash scripts/download_models.sh
PYTHONPATH=src python src/main.py   # Desktop app
# or
PYTHONPATH=src python src/web_app.py  # Web app → http://localhost:5000
```

---

## 2. Raspberry Pi 5 Setup

### Hardware Required

| Item | Specification |
|------|--------------|
| Raspberry Pi 5 | 8 GB RAM (recommended) |
| MicroSD Card | 64 GB+ Class 10 / A2 |
| Power Supply | Official RPi 5 USB-C PD 27W |
| Active Cooler | RPi 5 Active Cooler (required for sustained inference) |
| USB Camera | Any V4L2-compatible USB webcam |
| USB-A Hub | For camera + Pico W + keyboard |

### OS Installation

1. Download **Raspberry Pi OS Bookworm (64-bit)** from [raspberrypi.com/software](https://www.raspberrypi.com/software/)
2. Flash using Raspberry Pi Imager with:
   - Hostname: `waste-pi`
   - Enable SSH
   - Set username/password
   - Configure Wi-Fi (or use Ethernet)

### Software Setup on RPi 5

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies
sudo apt-get install -y \
  python3-pip python3-venv git \
  libglib2.0-0 libsm6 libxext6 libxrender-dev \
  libgomp1 v4l-utils libcamera-apps

# Clone the repository
git clone https://github.com/vskeerthan/bharat-aisoc-ps2.git
cd bharat-aisoc-ps2

# Run setup script
bash scripts/setup.sh

# Download Llama model (requires ~800 MB)
bash scripts/download_models.sh
```

### Systemd Service (Auto-start)

Create `/etc/systemd/system/bharat-aisoc-waste.service`:

```ini
[Unit]
Description=Bharat AI-SoC Waste Classification Web App
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/bharat-aisoc/src
ExecStart=/home/pi/bharat-aisoc/.venv/bin/python web_app.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable bharat-aisoc-waste.service
sudo systemctl start bharat-aisoc-waste.service
sudo journalctl -u bharat-aisoc-waste.service -f
```

### Performance Tuning

```bash
# Lock CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check temperature during inference (should stay < 80°C with active cooler)
vcgencmd measure_temp

# Increase swap for Llama loading
sudo dphys-swapfile swapoff
sudo sed -i 's/CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
sudo dphys-swapfile setup && sudo dphys-swapfile swapon
```

---

## 3. Raspberry Pi Pico W Sensor Array

### Hardware Required

| Component | Model | Quantity |
|-----------|-------|----------|
| MCU | Raspberry Pi Pico W | 1 |
| Ultrasonic | HC-SR04 | 1 |
| Temp/Humidity | DHT11 | 1 |
| Gas Sensor | MQ-135 | 1 |
| Moisture Sensor | Capacitive soil v1.2 | 1 |
| Resistor | 1 kΩ | 1 (HC-SR04 echo voltage divider) |
| Resistor | 2 kΩ | 1 (HC-SR04 echo voltage divider) |
| Breadboard | 400-point | 1 |
| Jumper Wires | M-M, M-F | 20+ |
| USB-A to Micro-USB | Data cable | 1 |

### Wiring Diagram

```
Raspberry Pi Pico W Pin Layout:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    ┌───────────┐
              GP0 ──┤ 1      40 ├── VBUS (5V from USB)
              GP1 ──┤ 2      39 ├── VSYS
              GND ──┤ 3      38 ├── GND
              GP2 ──┤ 4      37 ├── 3V3_EN
              GP3 ──┤ 5      36 ├── 3V3 OUT
              GP4 ──┤ 6      35 ├── ADC_VREF
              GP5 ──┤ 7      34 ├── GP28 / ADC2
              GND ──┤ 8      33 ├── GND / AGND
              GP6 ──┤ 9      32 ├── GP27 / ADC1  ←── Moisture Sensor (OUT)
              GP7 ──┤10      31 ├── GP26 / ADC0  ←── MQ-135 (AOUT)
              GP8 ──┤11      30 ├── RUN
              GP9 ──┤12      29 ├── GP22
              GND ──┤13      28 ├── GND
             GP10 ──┤14      27 ├── GP21
             GP11 ──┤15      26 ├── GP20
             GP12 ──┤16      25 ├── GP19
             GP13 ──┤17      24 ├── GP18
              GND ──┤18      23 ├── GND
             GP14 ──┤19      22 ├── GP17
             GP15 ──┤20      21 ├── GP16
                    └───────────┘

Sensor connections:
  HC-SR04 TRIG ──────────────► GP0 (pin 1)
  HC-SR04 ECHO ──[1kΩ]──────► GP1 (pin 2)
                       └──[2kΩ]──► GND  (voltage divider: 5V→3.3V)
  HC-SR04 VCC  ──────────────► VBUS (5V)
  HC-SR04 GND  ──────────────► GND

  DHT11 DATA   ──────────────► GP2 (pin 4)
  DHT11 VCC    ──────────────► 3V3 OUT (pin 36)
  DHT11 GND    ──────────────► GND

  MQ-135 AOUT  ──────────────► GP26 / ADC0 (pin 31)
  MQ-135 VCC   ──────────────► VBUS (5V)
  MQ-135 GND   ──────────────► GND
  (MQ-135 DOUT — not used)

  Moisture OUT ──────────────► GP27 / ADC1 (pin 32)
  Moisture VCC ──────────────► 3V3 OUT (pin 36)
  Moisture GND ──────────────► GND
```

### Flashing MicroPython Firmware

1. Hold **BOOTSEL** button on Pico W while connecting USB
2. A drive named `RPI-RP2` appears on your computer
3. Download MicroPython for Pico W: https://micropython.org/download/RPI_PICO_W/
4. Copy the `.uf2` file to the `RPI-RP2` drive (auto-flashes and reboots)
5. Upload the sensor firmware:

```bash
# Using mpremote (install: pip install mpremote)
cd src/pico_sensors
mpremote cp main.py :main.py
mpremote reset
```

Or use **Thonny IDE** (recommended for beginners):
- Open Thonny → Select interpreter → MicroPython (Raspberry Pi Pico)
- Open `src/pico_sensors/main.py` → Run / Save to Pico

### Verifying Pico W Data

```bash
# On host: listen to Pico W serial output
python3 -c "
import serial, json
s = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
while True:
    line = s.readline().decode().strip()
    if line:
        d = json.loads(line)
        print(f'Fill: {d[\"fill_pct\"]:.1f}%  Temp: {d[\"temp_c\"]:.1f}°C  Gas: {d[\"gas_ppm\"]:.0f}ppm')
"
```

---

## 4. Zynq-7000 FPGA Setup

### Supported Boards

| Board | Chip | Status |
|-------|------|--------|
| ZedBoard | XC7Z020-CLG484-1 | ✅ Supported |
| ZC702 | XC7Z020-CLG484-1 | ✅ Supported |

### Prerequisites

- Xilinx Vivado 2023.2+ (Design Suite or ML Edition)
- Xilinx SDK / Vitis 2023.2
- PetaLinux 2023.2 (for Linux-based PS)
- 32 GB RAM on host machine (Vivado synthesis)

### Project Structure

```
zynq-7000 soc/
├── vivado/          ← Vivado project files (.xpr, .bd)
├── bram/            ← BRAM controller HDL
└── external/        ← External IP cores

zc702/
├── vivado/          ← ZC702-specific Vivado project
├── bram/
└── external/
```

### Quick Start (ZedBoard)

```bash
# 1. Open Vivado
vivado zynq-7000\ soc/vivado/waste_classifier.xpr

# 2. Run synthesis and implementation
# Vivado GUI → Flow → Run Implementation

# 3. Export hardware (.xsa)
# File → Export Hardware → Include bitstream

# 4. Launch Vitis for software development
# Tools → Launch Vitis IDE

# 5. Program the board
# Xilinx → Program FPGA → Select .bit file
```

### AXI DMA Configuration

The PL (FPGA fabric) is configured with:
- **AXI DMA**: Scatter-gather mode for frame transfer
- **BRAM Controller**: 1 MB block RAM for frame buffering
- **AXI Interconnect**: Connects PS HP port to PL IP

See `zynq-7000 soc/vivado/` for the full block design and `docs/DEPLOYMENT.md` for the PetaLinux build flow.

---

## 5. Full System Wiring Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                     FULL SYSTEM CONNECTIONS                             │
│                                                                         │
│  USB Camera                                                             │
│  ┌──────────┐  USB-A                                                   │
│  │  Webcam  ├──────────────────────────────────► Raspberry Pi 5        │
│  └──────────┘                                    USB-A port             │
│                                                                         │
│  Raspberry Pi Pico W                                                    │
│  ┌──────────┐  USB (data + power)                                       │
│  │  Pico W  ├──────────────────────────────────► Raspberry Pi 5        │
│  │          │                                    USB-A port             │
│  │ Sensors: │                                                           │
│  │ HC-SR04  │                                                           │
│  │ DHT11    │                                                           │
│  │ MQ-135   │                                                           │
│  │ Moisture │                                                           │
│  └──────────┘                                                           │
│                                                                         │
│  Raspberry Pi 5                                                         │
│  ┌──────────────────────────────────────────┐                          │
│  │  • Runs: web_app.py / main.py            │  Ethernet / Wi-Fi        │
│  │  • YOLOv8 inference (CPU)                ├────────────────► Network │
│  │  • Llama 3.2 1B inference                │                 (browser) │
│  │  • SQLite database                       │                          │
│  │  • Flask web server :5000                │  HDMI (optional)         │
│  │  • USB: camera + Pico W                  ├────────────────► Monitor │
│  └──────────────────────────────────────────┘                          │
│                                                                         │
│  Zynq-7000 (optional FPGA acceleration)                                │
│  ┌──────────────────────────────────────────┐                          │
│  │  PS: ARM Cortex-A9 (same software stack)  │                         │
│  │  PL: AXI DMA + inference accelerator      │  UART / USB             │
│  │  BRAM: frame buffer                       ├─────────────► Host PC   │
│  └──────────────────────────────────────────┘  (Vivado/Vitis)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Power Requirements

| Component | Power |
|-----------|-------|
| Raspberry Pi 5 (8GB, full load) | ~12W |
| Raspberry Pi Pico W | ~0.5W |
| USB Camera | ~1.5W |
| HC-SR04 | ~15mA @ 5V = 0.075W |
| DHT11 | ~2.5mA @ 3.3V = 0.008W |
| MQ-135 (after warmup) | ~135mA @ 5V = 0.675W |
| Capacitive Moisture | ~5mA @ 3.3V = 0.0165W |
| **Total (RPi 5 + peripherals)** | **~15W** |
| Zynq-7000 ZedBoard | ~10–15W |

**Recommended power supplies:**
- RPi 5: Official 27W USB-C PD power supply
- ZedBoard: 12V / 3A barrel jack

