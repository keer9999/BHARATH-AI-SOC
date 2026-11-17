#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/download_models.sh — Download & Verify AI model files
# Downloads the Llama 3.2 1B GGUF model and verifies YOLO detection weights.
# Usage: bash scripts/download_models.sh [--llama] [--yolo] [--all]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}→ $*${RESET}"; }
success() { echo -e "${GREEN}✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
error()   { echo -e "${RED}✗ $*${RESET}" >&2; exit 1; }

# ─── Configuration ────────────────────────────────────────────────────────────
LLAMA_MODEL_NAME="llama-3.2-1b-instruct-q4_k_m.gguf"
LLAMA_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
MODEL_DIR="src/models"
ROOT_MODEL_DIR="models"

DO_LLAMA=true
DO_YOLO=true

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --llama) DO_LLAMA=true; DO_YOLO=false; shift ;;
    --yolo)  DO_YOLO=true; DO_LLAMA=false; shift ;;
    --all)   DO_LLAMA=true; DO_YOLO=true; shift ;;
    --model) LLAMA_MODEL_NAME="$2"; shift 2 ;;
    --dir)   MODEL_DIR="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/download_models.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --all       Download / verify all models (Llama 3.2 + YOLO) [default]"
      echo "  --llama     Download Llama 3.2 GGUF only"
      echo "  --yolo      Verify & sync YOLO weights only"
      echo "  --model <f> Specify custom Llama GGUF filename"
      echo "  --dir <p>   Specify target directory for models"
      exit 0 ;;
    *)
      shift ;;
  esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Bharat AI-SoC PS2 — Model Manager & Downloader      ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ─── 1. YOLO Verification & Sync ─────────────────────────────────────────────
if [ "$DO_YOLO" = true ]; then
  info "Verifying YOLO models..."
  mkdir -p "$ROOT_MODEL_DIR" "src/yolo/weights" "hardware/zynq7000/external/models" "hardware/zc702/external/models"

  # Sync best.pt
  if [ -f "$ROOT_MODEL_DIR/best.pt" ]; then
    success "Found $ROOT_MODEL_DIR/best.pt"
    [ -f "src/yolo/weights/best.pt" ] || cp "$ROOT_MODEL_DIR/best.pt" "src/yolo/weights/best.pt"
    [ -f "hardware/zynq7000/external/models/best.pt" ] || cp "$ROOT_MODEL_DIR/best.pt" "hardware/zynq7000/external/models/best.pt"
    [ -f "hardware/zc702/external/models/best.pt" ] || cp "$ROOT_MODEL_DIR/best.pt" "hardware/zc702/external/models/best.pt"
  elif [ -f "src/yolo/weights/best.pt" ]; then
    success "Found src/yolo/weights/best.pt"
    cp "src/yolo/weights/best.pt" "$ROOT_MODEL_DIR/best.pt"
  else
    warn "best.pt not found. Run 'python src/train_waste_model.py' to train custom weights."
  fi

  # Sync ecosort_waste_yolo11s.pt (66-class granular waste model)
  if [ -f "src/yolo/weights/ecosort_waste_yolo11s.pt" ]; then
    success "Found src/yolo/weights/ecosort_waste_yolo11s.pt"
    [ -f "$ROOT_MODEL_DIR/ecosort_waste_yolo11s.pt" ] || cp "src/yolo/weights/ecosort_waste_yolo11s.pt" "$ROOT_MODEL_DIR/ecosort_waste_yolo11s.pt"
    [ -f "hardware/zynq7000/external/models/ecosort_waste_yolo11s.pt" ] || cp "src/yolo/weights/ecosort_waste_yolo11s.pt" "hardware/zynq7000/external/models/ecosort_waste_yolo11s.pt"
  elif [ -f "$ROOT_MODEL_DIR/ecosort_waste_yolo11s.pt" ]; then
    success "Found $ROOT_MODEL_DIR/ecosort_waste_yolo11s.pt"
    cp "$ROOT_MODEL_DIR/ecosort_waste_yolo11s.pt" "src/yolo/weights/ecosort_waste_yolo11s.pt"
  fi

  # Sync weights.pt (bottle specialist model)
  if [ -f "src/yolo/weights/weights.pt" ]; then
    success "Found src/yolo/weights/weights.pt"
    [ -f "$ROOT_MODEL_DIR/weights.pt" ] || cp "src/yolo/weights/weights.pt" "$ROOT_MODEL_DIR/weights.pt"
  elif [ -f "$ROOT_MODEL_DIR/weights.pt" ]; then
    success "Found $ROOT_MODEL_DIR/weights.pt"
    cp "$ROOT_MODEL_DIR/weights.pt" "src/yolo/weights/weights.pt"
  fi

  # Sync yolov8n.pt
  if [ -f "$ROOT_MODEL_DIR/yolov8n.pt" ]; then
    success "Found $ROOT_MODEL_DIR/yolov8n.pt"
    [ -f "src/yolov8n.pt" ] || cp "$ROOT_MODEL_DIR/yolov8n.pt" "src/yolov8n.pt"
    [ -f "hardware/zynq7000/external/models/yolov8n.pt" ] || cp "$ROOT_MODEL_DIR/yolov8n.pt" "hardware/zynq7000/external/models/yolov8n.pt"
    [ -f "hardware/zc702/external/models/yolov8n.pt" ] || cp "$ROOT_MODEL_DIR/yolov8n.pt" "hardware/zc702/external/models/yolov8n.pt"
  elif [ -f "src/yolov8n.pt" ]; then
    success "Found src/yolov8n.pt"
    cp "src/yolov8n.pt" "$ROOT_MODEL_DIR/yolov8n.pt"
  else
    info "Downloading yolov8n.pt from GitHub releases..."
    curl -L -s -o "$ROOT_MODEL_DIR/yolov8n.pt" "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt" || true
    if [ -f "$ROOT_MODEL_DIR/yolov8n.pt" ]; then
      cp "$ROOT_MODEL_DIR/yolov8n.pt" "src/yolov8n.pt"
      success "Downloaded yolov8n.pt"
    fi
  fi
fi

# ─── 2. Llama 3.2 GGUF Download ──────────────────────────────────────────────
if [ "$DO_LLAMA" = true ]; then
  MODEL_PATH="$MODEL_DIR/$LLAMA_MODEL_NAME"

  if [ -f "$MODEL_PATH" ]; then
    success "Llama model already exists at $MODEL_PATH"
  else
    mkdir -p "$MODEL_DIR"
    info "Target Llama model: $MODEL_PATH"

    DOWNLOAD_CMD=""
    if command -v huggingface-cli &>/dev/null; then
      DOWNLOAD_CMD="hf"
    elif command -v wget &>/dev/null; then
      DOWNLOAD_CMD="wget"
    elif command -v curl &>/dev/null; then
      DOWNLOAD_CMD="curl"
    else
      error "No download tool found. Install one of: huggingface-cli, wget, curl"
    fi

    HF_URL="https://huggingface.co/${LLAMA_REPO}/resolve/main/${LLAMA_MODEL_NAME}"
    info "Downloading Llama 3.2 1B Instruct GGUF from: $HF_URL"

    case "$DOWNLOAD_CMD" in
      hf)
        huggingface-cli download "$LLAMA_REPO" "$LLAMA_MODEL_NAME" --local-dir "$MODEL_DIR"
        ;;
      wget)
        wget --show-progress -c -O "$MODEL_PATH" "$HF_URL"
        ;;
      curl)
        curl -L --progress-bar -o "$MODEL_PATH" "$HF_URL"
        ;;
    esac

    if [ -f "$MODEL_PATH" ]; then
      SIZE_MB=$(($(stat -c%s "$MODEL_PATH") / 1024 / 1024))
      success "Model downloaded successfully: $MODEL_PATH (${SIZE_MB} MB)"
    else
      error "Download failed — file not found at $MODEL_PATH"
    fi
  fi
fi

echo ""
success "All required AI model checks complete!"
