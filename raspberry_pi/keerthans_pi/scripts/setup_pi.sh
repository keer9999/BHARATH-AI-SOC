#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/setup_pi.sh  —  One-shot setup for Keerthan's Raspberry Pi
#
# Run as:  bash raspberry_pi/keerthans_pi/scripts/setup_pi.sh
# Tested on:  Raspberry Pi 4B / 5  (64-bit Raspberry Pi OS Bookworm)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PI_DIR/../.." && pwd)"

echo "════════════════════════════════════════════════════════"
echo "  EcoSort AI — Keerthan's Pi Setup"
echo "  Repo: $REPO_ROOT"
echo "════════════════════════════════════════════════════════"

# ── 1. System packages ────────────────────────────────────────────────────────
echo ""
echo "▶ [1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3-pip python3-venv python3-dev \
    libopencv-dev libgl1 libglib2.0-0 \
    build-essential cmake git \
    libgomp1 \
    udev curl wget

# ── 2. Python virtual environment ────────────────────────────────────────────
echo ""
echo "▶ [2/6] Creating Python venv at $PI_DIR/.venv ..."
python3 -m venv "$PI_DIR/.venv"
source "$PI_DIR/.venv/bin/activate"

# ── 3. Python packages ────────────────────────────────────────────────────────
echo ""
echo "▶ [3/6] Installing Python packages (this takes a few minutes)..."
pip install --upgrade pip --quiet
pip install -r "$PI_DIR/requirements-pi.txt" --quiet

echo "   ✔ onnxruntime, ultralytics, flask, llama-cpp-python installed"

# ── 4. Verify ONNX model ─────────────────────────────────────────────────────
echo ""
echo "▶ [4/6] Checking ONNX model..."
ONNX_PATH="$REPO_ROOT/src/yolo/weights/ecosort_waste_yolo11s.onnx"
if [ -f "$ONNX_PATH" ]; then
    SIZE=$(du -sh "$ONNX_PATH" | cut -f1)
    echo "   ✔ Found: ecosort_waste_yolo11s.onnx ($SIZE)"
else
    echo "   ✗ ONNX model not found at $ONNX_PATH"
    echo "     Make sure the repo was pulled with git-lfs or the file is tracked."
    echo "     Run: git lfs pull   (if using git-lfs)"
    exit 1
fi

# ── 5. Download Llama GGUF model ─────────────────────────────────────────────
echo ""
echo "▶ [5/6] Checking Llama GGUF model..."
LLAMA_DIR="$PI_DIR/models"
LLAMA_FILE="$LLAMA_DIR/llama-3.2-1b-instruct-q4_k_m.gguf"
mkdir -p "$LLAMA_DIR"

if [ -f "$LLAMA_FILE" ]; then
    SIZE=$(du -sh "$LLAMA_FILE" | cut -f1)
    echo "   ✔ Found: llama-3.2-1b-instruct-q4_k_m.gguf ($SIZE)"
else
    echo "   Downloading Llama 3.2 1B Instruct Q4_K_M (~750 MB) ..."
    wget -q --show-progress \
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf" \
        -O "$LLAMA_FILE" || {
        echo "   ✗ Download failed. Try manually:"
        echo "     wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf -O $LLAMA_FILE"
    }
fi

# ── 6. Quick smoke test ───────────────────────────────────────────────────────
echo ""
echo "▶ [6/6] Running smoke test..."
cd "$REPO_ROOT/src"
python3 - <<'EOF'
import os, sys
sys.path.insert(0, ".")
os.environ["ECOSORT_ONNX_ONLY"] = "1"
os.environ["ECOSORT_CUSTOM_MODEL"] = os.path.abspath("yolo/weights/ecosort_waste_yolo11s.onnx")

from config import DetectionConfig
cfg = DetectionConfig()
print(f"   model_path       : {cfg.model_path}")
print(f"   custom_model_path: {cfg.custom_model_path}")

try:
    from waste_detector import WasteDetector
    det = WasteDetector(cfg)
    ok = det.load_model()
    print(f"   load_model()     : {'✔ OK' if ok else '✗ FAILED'}")
    print(f"   Classes          : {len(det.custom_class_names)}")
except Exception as e:
    print(f"   ✗ Error: {e}")
EOF

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Setup complete!  To start the web app:"
echo ""
echo "  cd raspberry_pi/keerthans_pi"
echo "  source .venv/bin/activate"
echo "  source config/pi_config.env"
echo "  python ../../src/web_app.py --host 0.0.0.0 --port 5001"
echo ""
echo "  Then open: http://<pi-ip-address>:5001"
echo "════════════════════════════════════════════════════════"
