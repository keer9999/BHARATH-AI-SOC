# Keerthan's Pi — EcoSort AI Deployment Guide

> **Hardware:** Raspberry Pi 4B / 5 · 64-bit Raspberry Pi OS Bookworm  
> **Model:** `ecosort_waste_yolo11s.onnx` — unified 66-class waste detector  
> **LLM:** Llama 3.2 1B Instruct (GGUF via llama-cpp-python **or** ExecuTorch .pte)

---

## Folder structure

```
raspberry_pi/keerthans_pi/
├── config/
│   └── pi_config.env          # Environment variables (model paths, ports)
├── docs/
│   └── EXECUTORCH_SETUP.md    # Optional: compile Llama .pte for ARM
├── models/                    # Created by setup_pi.sh (not in git)
│   └── llama-3.2-1b-instruct-q4_k_m.gguf
├── scripts/
│   ├── setup_pi.sh            # ONE-SHOT setup: system deps + venv + download
│   ├── start_pi.sh            # Start the web app
│   └── download_llama.sh      # Download Llama GGUF separately
├── requirements-pi.txt        # Pi-specific Python deps (onnxruntime, no torch)
└── README.md                  # You are here
```

The ONNX model lives in the main repo at:  
`src/yolo/weights/ecosort_waste_yolo11s.onnx` (37 MB, 66 waste classes)

---

## Quick start (3 commands)

```bash
# 1. Clone the repo (or git pull)
git clone https://github.com/<your-org>/ecosort-ai.git
cd ecosort-ai

# 2. Run the one-shot setup script
bash raspberry_pi/keerthans_pi/scripts/setup_pi.sh

# 3. Start the web app
bash raspberry_pi/keerthans_pi/scripts/start_pi.sh
# Then open: http://<pi-ip-address>:5001
```

---

## What runs on the Pi

| Component | File | Backend |
|-----------|------|---------|
| **Waste detection** | `ecosort_waste_yolo11s.onnx` | ONNX Runtime (CPU) |
| **AI disposal advice** | `llama-3.2-1b-instruct-q4_k_m.gguf` | llama-cpp-python |
| **Alternative LLM** | `llama3_2_1b.pte` *(optional)* | ExecuTorch ARM |
| **Web dashboard** | Flask `web_app.py` | Python 3.11 |
| **Sensor data** | Pico W via `/dev/ttyACM0` | pyserial |

---

## Model details — ONNX (66 classes)

```
File   : src/yolo/weights/ecosort_waste_yolo11s.onnx
Size   : ~37 MB
Input  : [1, 3, 640, 640]  float32
Output : [1, 70, 8400]     float32
Opset  : 12
Speed  : ~300–500 ms/frame on Pi 4B CPU
```

**All 66 waste classes covered:**  
Plastics (bottles, bags, containers, film, foam, PET/HDPE/PP/PS/PVC resin codes),  
Glass (clear, brown, green, broken), Metal (aluminium cans, food cans, steel),  
Paper & cardboard, Organic & food waste, Hazardous (batteries, e-waste,  
chemicals), Textiles.

> **Why ONNX instead of PyTorch?**  
> `onnxruntime` starts in ~1 s and uses ~120 MB RAM.  
> Loading the same model via `torch` / `ultralytics` takes ~600 MB on Pi — often  
> causes an OOM kill before the first inference.

---

## Llama 3.2 1B — GGUF backend (default)

The GGUF model is downloaded by `setup_pi.sh`. It runs fully on-device with
no internet and powers the AI waste disposal advice panel in the web UI.

```
Model  : Llama-3.2-1B-Instruct-Q4_K_M.gguf
Size   : ~750 MB
RAM    : ~700 MB at peak
Speed  : ~3–8 tokens/sec on Pi 4B (no GPU)
Backend: llama-cpp-python  (ARM NEON optimised)
```

> **Tips for speed:**  
> - Use `Q4_K_M` quantization (already the default — good accuracy/speed ratio)  
> - Add `export LLAMA_N_THREADS=4` in `config/pi_config.env`  
> - If Pi 5: `Q5_K_M` gives noticeably better quality at ~same speed

---

## Llama 3.2 1B — ExecuTorch .pte backend (optional, faster)

ExecuTorch is Meta's on-device inference framework optimised for ARM.  
It can run Llama 3.2 1B at ~2× the speed of GGUF on a Pi 5 with the  
`xnnpack` backend.

> **Full instructions:** See [`docs/EXECUTORCH_SETUP.md`](docs/EXECUTORCH_SETUP.md)

Short version:
```bash
# After following EXECUTORCH_SETUP.md, place these files:
cp llama3_2_1b.pte  raspberry_pi/keerthans_pi/models/
cp tokenizer.model  raspberry_pi/keerthans_pi/models/

# Switch backend in config/pi_config.env
export LLAMA_BACKEND="executorch"
export LLAMA_PTE_PATH="$(pwd)/models/llama3_2_1b.pte"
export LLAMA_TOKENIZER_PATH="$(pwd)/models/tokenizer.model"
```

---

## Running the app manually (without the scripts)

```bash
cd raspberry_pi/keerthans_pi
source .venv/bin/activate
source config/pi_config.env

# From repo root/src:
cd ../../src
python web_app.py --host 0.0.0.0 --port 5001
```

---

## Run as a systemd service (auto-start on boot)

```bash
sudo tee /etc/systemd/system/ecosort.service > /dev/null <<EOF
[Unit]
Description=EcoSort AI — Waste Segregation System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ecosort-ai
ExecStart=/bin/bash /home/pi/ecosort-ai/raspberry_pi/keerthans_pi/scripts/start_pi.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ecosort
sudo systemctl start ecosort
sudo journalctl -u ecosort -f   # View logs
```

---

## Expected performance on Raspberry Pi 4B (4 GB)

| Metric | Value |
|--------|-------|
| ONNX model load time | ~2 s |
| Inference per frame | ~300–500 ms |
| Llama GGUF load time | ~8–12 s |
| Llama tokens/sec | ~3–8 tok/s |
| Total RAM at runtime | ~900 MB – 1.1 GB |
| Recommended Pi RAM | **4 GB** (8 GB for comfort) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Killed` / OOM during startup | Increase Pi swap: `sudo dphys-swapfile swapconf` → `CONF_SWAPSIZE=2048` |
| `ModuleNotFoundError: onnxruntime` | `pip install onnxruntime` in the venv |
| ONNX model not found | Check `git lfs pull` or manually copy from `src/yolo/weights/` |
| Llama GGUF not found | Run `bash scripts/download_llama.sh` |
| Serial port not found | Check `ls /dev/ttyACM*`; update `SERIAL_PORT` in `pi_config.env` |
| Web app not accessible | Allow port: `sudo ufw allow 5001/tcp` |

---

## Connecting Pico W sensors

The Pico W connects via USB serial (`/dev/ttyACM0` by default).  
Sensors wired up:

| Sensor | Pico Pin | Purpose |
|--------|----------|---------|
| HC-SR04 | GP2 (echo) / GP3 (trig) | Bin fill level |
| DHT11 | GP15 | Temperature & humidity (wet waste) |
| MQ gas sensor | GP26 (ADC0) / GP16 (DO) | Hazardous gas detection |
| HW-103 moisture | GP27 (ADC1) | Wet vs dry waste |

Flash the Pico with the firmware at `src/pico/pico_sensor.py` using Thonny  
or `mpremote copy src/pico/pico_sensor.py :main.py`.

---

*Generated for Keerthan's Raspberry Pi — EcoSort AI v2.0 · November 2026*
