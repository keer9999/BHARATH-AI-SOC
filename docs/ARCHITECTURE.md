# System Architecture — Bharat AI-SoC PS2

## Overview

The Bharat AI-SoC PS2 Waste Classification System is a layered, edge-first architecture that runs all AI inference on-device without any cloud dependency. The system is composed of four logical layers: **Sensing**, **Processing**, **Acceleration**, and **Presentation**.

---

## Layer 1 — Sensing

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SENSING LAYER                                │
│                                                                      │
│   ┌──────────────────┐         ┌─────────────────────────────────┐  │
│   │   Camera Module  │         │     Raspberry Pi Pico W          │  │
│   │   (USB / CSI)    │         │       (MicroPython)              │  │
│   │   1080p / 720p   │         │                                  │  │
│   │   30 FPS capture │         │  ┌─────────┐    ┌────────────┐  │  │
│   └────────┬─────────┘         │  │ HC-SR04 │    │   DHT11    │  │  │
│            │ OpenCV frames      │  │Ultrasn. │    │ Temp/Hum   │  │  │
│            │ (BGR numpy array)  │  │GPIO 0,1 │    │ GPIO 2     │  │  │
│            │                   │  └─────────┘    └────────────┘  │  │
│            │                   │  ┌─────────┐    ┌────────────┐  │  │
│            │                   │  │ MQ-135  │    │ Capacitive │  │  │
│            │                   │  │Gas/VOC  │    │ Moisture   │  │  │
│            │                   │  │ADC0     │    │ ADC1       │  │  │
│            │                   │  └─────────┘    └────────────┘  │  │
│            │                   │                                  │  │
│            │                   │  USB-CDC JSON stream @ 115200    │  │
│            │                   └───────────────┬──────────────────┘  │
└────────────┼───────────────────────────────────┼────────────────────┘
             │ Frame                              │ Sensor JSON
             ▼                                   ▼
```

### Camera

- **Interface**: USB (V4L2) or CSI (Raspberry Pi Camera Module)
- **Resolution**: 1920×1080 or 1280×720 (auto-negotiated)
- **Frame rate**: 30 FPS capture → processed at ≥15 FPS detection rate
- **OpenCV capture**: `cv2.VideoCapture(device_id)`

### Raspberry Pi Pico W Sensors

| Sensor | Model | GPIO / ADC | Data |
|--------|-------|------------|------|
| Ultrasonic | HC-SR04 | GPIO 0 (TRIG), GPIO 1 (ECHO) | Distance (cm) |
| Temperature/Humidity | DHT11 | GPIO 2 | Temp (°C), RH (%) |
| Gas / VOC | MQ-135 | ADC0 (GPIO 26) | Analog voltage → ppm |
| Moisture | Capacitive soil | ADC1 (GPIO 27) | Analog → % moisture |

**Protocol**: Pico W sends newline-delimited JSON over USB-CDC at 115200 baud:
```json
{"timestamp": 1737123456, "distance_cm": 23.4, "temp_c": 28.1, "humidity_pct": 65.2, "gas_ppm": 142.3, "moisture_pct": 18.7, "fill_pct": 46.8}
```

---

## Layer 2 — Processing (Raspberry Pi 5)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER — Raspberry Pi 5                    │
│                                                                         │
│   webcam_capture.py          data_fusion.py                            │
│   ┌───────────────────┐      ┌──────────────────────────────────────┐  │
│   │  Camera Thread    │      │           Data Fusion Engine          │  │
│   │  cv2.VideoCapture │      │                                       │  │
│   │  Frame queue      │─────►│  vision_detections + sensor_reading  │  │
│   └───────────────────┘      │  → fused_event (class + fill + risk) │  │
│                               └───────────────┬──────────────────────┘  │
│   waste_detector.py                           │                         │
│   ┌───────────────────┐                       │                         │
│   │  YOLOv8 Engine    │                       │                         │
│   │  model: yolov8n   │                       ▼                         │
│   │  5 classes        │      ┌──────────────────────────────────────┐  │
│   │  conf ≥ 0.5       │─────►│          database.py (SQLite)        │  │
│   │  ≥15 FPS on RPi5  │      │  detections, sessions, sensor_logs   │  │
│   └───────────────────┘      └───────────────┬──────────────────────┘  │
│                                               │                         │
│   llama_engine.py                             │                         │
│   ┌───────────────────┐      alert_system.py  │                         │
│   │  Llama 3.2 1B     │      ┌──────────────┐ │  report_generator.py   │
│   │  GGUF Q4_K_M      │      │ Email/SMS    │ │  ┌──────────────────┐  │
│   │  llama-cpp-python │      │ Alerts       │ │  │  PDF/CSV Reports │  │
│   │  Disposal advice  │      └──────────────┘ │  │  ReportLab/MPL   │  │
│   └───────────────────┘                       │  └──────────────────┘  │
└───────────────────────────────────────────────┼───────────────────────┘
                                                 │ Flask REST API
                                                 ▼
```

### Module Responsibilities

| Module | File | Responsibility |
|--------|------|----------------|
| Camera capture | `webcam_capture.py` | Thread-safe frame acquisition |
| Waste detector | `waste_detector.py` | YOLOv8 inference, NMS, class mapping |
| LLM engine | `llama_engine.py` | Llama 3.2 1B GGUF inference |
| ExecuTorch LLM | `executorch_llama_engine.py` | ARM-optimised ExecuTorch inference |
| Data fusion | `data_fusion.py` | Multi-modal sensor+vision fusion |
| Database | `database.py` | SQLite CRUD, session management |
| Alerts | `alert_system.py` | Threshold-based email/SMS alerts |
| Reports | `report_generator.py` | PDF + CSV report generation |
| Config | `config.py` | Centralised configuration |

### Waste Classification Taxonomy

```
Waste
├── Plastic    → Recycle (blue bin)
├── Glass      → Recycle (green bin)  
├── Metal      → Recycle (yellow bin)
├── Paper      → Recycle (blue bin)
└── Organic    → Compost (brown bin)
```

### Data Fusion Logic

```python
# Pseudo-code for fused_event calculation
fill_pct = 100 - (distance_cm / BIN_HEIGHT_CM * 100)
organic_risk = moisture_pct > 40 and gas_ppm > 200 and temp_c > 30
overfill_alert = fill_pct > FILL_THRESHOLD  # default: 80%
fused_event = FusedEvent(
    detected_class=yolo_class,
    confidence=yolo_conf,
    fill_pct=fill_pct,
    organic_risk=organic_risk,
    overfill_alert=overfill_alert,
    sensor_reading=sensor_json
)
```

---

## Layer 3 — Acceleration (Zynq-7000)

```
┌────────────────────────────────────────────────────────────────────────┐
│                 ACCELERATION LAYER — Zynq-7000 SoC                      │
│                                                                         │
│   ┌──────────────────────────┐  AXI Interconnect                       │
│   │   PS — ARM Cortex-A9     │◄──────────────────────────────────────► │
│   │   • Petalinux / RTOS     │                                         │
│   │   • OpenCV preprocessing │  ┌────────────────────────────────────┐ │
│   │   • YOLOv8 postprocess   │  │   PL — Programmable Logic          │ │
│   │   • Sensor UART / USB    │  │   • AXI DMA engine                 │ │
│   │   • Flask web server     │  │   • BRAM frame buffer (1 MB)       │ │
│   │   • SQLite database      │  │   • Custom inference accelerator   │ │
│   └──────────────────────────┘  │   • VGA / HDMI output (optional)  │ │
│                                  └────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### AXI DMA Flow

```
Camera frame (PS DDR) 
  → AXI HP port 
  → AXI DMA MM2S 
  → PL inference accelerator (BRAM)
  → AXI DMA S2MM 
  → Result buffer (PS DDR)
  → YOLOv8 postprocessing (PS ARM)
```

---

## Layer 4 — Presentation

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PRESENTATION LAYER                              │
│                                                                      │
│   ┌───────────────────────────────────┐                             │
│   │          Flask Web App            │                             │
│   │          web_app.py               │                             │
│   │                                   │                             │
│   │  GET  /               Dashboard   │                             │
│   │  GET  /video_feed     MJPEG stream│ ◄── browser / mobile       │
│   │  GET  /api/status     JSON status │                             │
│   │  GET  /api/detections Detections  │                             │
│   │  GET  /api/stats      Statistics  │                             │
│   │  POST /api/chat       LLM chat    │                             │
│   │  GET  /api/report/pdf PDF report  │                             │
│   │  GET  /api/report/csv CSV export  │                             │
│   └───────────────────────────────────┘                             │
│                                                                      │
│   ┌───────────────────────────────────┐                             │
│   │       OpenCV Desktop App          │                             │
│   │       main.py                     │                             │
│   │                                   │                             │
│   │  Live camera window with HUD:     │                             │
│   │  • Bounding boxes + class labels  │ ◄── local display          │
│   │  • Confidence scores              │                             │
│   │  • Sensor readings panel          │                             │
│   │  • Fill-level progress bar        │                             │
│   │  • Alert popup overlays           │                             │
│   └───────────────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
Camera ─► OpenCV Frame ─► waste_detector.py (YOLOv8) ─────────────────┐
                                                                        │
Pico W ─► UART JSON ──► data_fusion.py ─────────────────────────────► fused_event
                                                                        │
                                                             ┌──────────┘
                                                             │
                              ┌──────────────────────────────▼─────────┐
                              │           database.py                   │
                              │  INSERT detection, session, sensor_log  │
                              └──────────┬──────────────────────────────┘
                                         │
                     ┌───────────────────┼──────────────────────┐
                     │                   │                       │
                     ▼                   ▼                       ▼
             llama_engine.py      alert_system.py        report_generator.py
             Disposal advice      Email/SMS alert         PDF / CSV report
                     │
                     ▼
              web_app.py / main.py
              (REST API / HUD)
```

---

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Object detection | YOLOv8n/s | Best speed/accuracy trade-off at edge |
| LLM | Llama 3.2 1B GGUF | Smallest capable model, runs on RPi 5 |
| LLM runtime | llama-cpp-python | GGUF quantization, no GPU required |
| Future LLM runtime | ExecuTorch | ARM-optimised, Arm's recommended path |
| Web framework | Flask | Lightweight, easy MJPEG streaming |
| Database | SQLite | Zero-config, file-based, sufficient for edge |
| FPGA | Zynq-7000 | PS+PL co-design, Arm cortex + FPGA fabric |
| Container | Docker | Reproducible deployment |

