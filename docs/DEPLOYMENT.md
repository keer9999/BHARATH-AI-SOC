# Deployment Guide

## Deployment Targets

| Target | Method | Use Case |
|--------|--------|----------|
| Raspberry Pi 5 | Native systemd | Primary edge deployment |
| Any Linux PC | Native venv | Development / demo |
| Docker (any host) | Docker Compose | Containerised dev/test |
| Zynq-7000 | PetaLinux | FPGA-accelerated deployment |

---

## 1. Raspberry Pi 5 — Native Deployment

### Prerequisites
- Raspberry Pi OS Bookworm (64-bit) installed
- SSH access enabled
- Internet connection for initial setup

### Quick Deploy

```bash
# On host machine — push code to RPi 5
rsync -avz --exclude='.git' --exclude='.venv' --exclude='*.gguf' \
  ./ pi@raspberry-pi-5.local:/home/pi/bharat-aisoc/

# SSH into RPi 5
ssh pi@raspberry-pi-5.local

# On RPi 5
cd /home/pi/bharat-aisoc
bash scripts/setup.sh
bash scripts/download_models.sh
```

### Systemd Auto-start

```bash
sudo cp scripts/bharat-aisoc-waste.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bharat-aisoc-waste.service
sudo systemctl start bharat-aisoc-waste.service

# Check status
sudo systemctl status bharat-aisoc-waste.service
sudo journalctl -u bharat-aisoc-waste.service -f
```

**Service file** (`/etc/systemd/system/bharat-aisoc-waste.service`):
```ini
[Unit]
Description=Bharat AI-SoC Waste Classification System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/bharat-aisoc/version one c2s/module
ExecStart=/home/pi/bharat-aisoc/.venv/bin/python web_app.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy (Optional)

For production exposure with TLS:

```nginx
# /etc/nginx/sites-available/waste-classifier
server {
    listen 80;
    server_name waste-pi.local;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Required for MJPEG streaming
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/waste-classifier /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### GitHub Actions Auto-deploy

Configure secrets in GitHub → Settings → Secrets:
- `RPI_HOST` — RPi 5 IP or hostname
- `RPI_USER` — SSH user (e.g. `pi`)
- `RPI_SSH_KEY` — Private key (generate with `ssh-keygen -t ed25519 -C "github-deploy"`)
- `RPI_DEPLOY_PATH` — e.g. `/home/pi/bharat-aisoc`

Trigger from GitHub Actions → Deploy to Raspberry Pi 5 → Run workflow.

---

## 2. Docker Deployment

### Prerequisites
- Docker 24.0+
- Docker Compose v2.20+

### Build & Run

```bash
# Build image
docker compose build

# Start web app + dashboard
docker compose up -d
docker compose --profile dashboard up -d  # Include Streamlit dashboard

# View logs
docker compose logs -f waste-web

# Stop
docker compose down
```

### Production Docker Config

For production, override settings in `docker-compose.override.yml`:

```yaml
# docker-compose.override.yml (not committed)
services:
  waste-web:
    environment:
      - FLASK_ENV=production
    restart: always
    deploy:
      resources:
        limits:
          memory: 6G
```

### Docker on Raspberry Pi 5

```bash
# Install Docker on RPi 5
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker pi
newgrp docker

# Build ARM64 image (may take 20+ min on RPi 5)
docker compose build

# Run
docker compose up -d
```

---

## 3. Zynq-7000 (PetaLinux) Deployment

### Overview

The Zynq-7000 deployment runs the same Python application stack on the ARM Cortex-A9 Processing System (PS), with the FPGA Programmable Logic (PL) providing optional inference acceleration.

### PetaLinux Build Flow

```bash
# 1. Export hardware from Vivado (.xsa file)
# Vivado → File → Export Hardware → Include bitstream → save as waste_classifier.xsa

# 2. Create PetaLinux project
petalinux-create --type project --template zynq --name waste-petalinux
cd waste-petalinux

# 3. Import hardware
petalinux-config --get-hw-description ../zynq-7000\ soc/vivado/waste_classifier.xsa

# 4. Configure rootfs — add Python3, pip, OpenCV
petalinux-config -c rootfs
# Enable: python3, python3-pip, python3-opencv, v4l-utils

# 5. Build
petalinux-build

# 6. Create boot files
petalinux-package --boot --fsbl images/linux/zynq_fsbl.elf \
                           --fpga images/linux/system.bit \
                           --u-boot --force

# 7. Flash SD card
# Copy BOOT.BIN, boot.scr, image.ub to FAT32 partition
# Copy rootfs.tar.gz to ext4 partition and extract
```

### Running on Zynq-7000

```bash
# Boot the board, SSH in (or use UART console)
ssh root@192.168.1.100

# Install Python deps
pip3 install flask ultralytics

# Download model
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Run web app
cd /home/root/bharat-aisoc/"version one c2s/module"
python3 web_app.py
```

---

## 4. Desktop / Development Deployment

```bash
# Clone
git clone https://github.com/vskeerthan/bharat-aisoc-ps2.git
cd bharat-aisoc-ps2

# Setup
bash scripts/setup.sh
bash scripts/download_models.sh

# Run desktop app
source .venv/bin/activate
cd "version one c2s/module"
python main.py

# Run web app
python web_app.py
# → http://localhost:5000
```

---

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_DEVICE` | `0` | OpenCV camera device index |
| `PICO_SERIAL_PORT` | `auto` | Serial port for Pico W (`/dev/ttyACM0` on Linux) |
| `WASTE_MODEL_PATH` | `models/` | Directory containing GGUF model files |
| `WASTE_DB_PATH` | `database/waste_data.db` | SQLite database path |
| `WASTE_REPORTS_PATH` | `reports/` | Report output directory |
| `WASTE_LOG_PATH` | `logs/waste_system.log` | Log file path |
| `FLASK_PORT` | `5000` | Flask server port |
| `YOLO_MODEL` | `yolov8n.pt` | YOLOv8 model variant |
| `YOLO_CONF_THRESHOLD` | `0.5` | Minimum detection confidence |
| `FILL_ALERT_THRESHOLD` | `80` | Bin fill % to trigger alert |

Set via `.env` file in the module directory or as system environment variables.

---

## 6. Monitoring

### Log Files

```bash
# Application logs
tail -f "version one c2s/module/logs/waste_system.log"

# Docker logs
docker compose logs -f waste-web

# Systemd logs
sudo journalctl -u bharat-aisoc-waste.service -f
```

### Health Check

```bash
# Check API health
curl http://localhost:5000/api/status | python3 -m json.tool

# Check fill level
curl -s http://localhost:5000/api/status | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'Fill: {d[\"sensors\"][\"fill_pct\"]:.1f}%')
print(f'FPS: {d[\"camera\"][\"fps\"]:.1f}')
"
```
