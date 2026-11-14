#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/setup.sh — Bharat AI-SoC PS2 Automated Setup
# Creates a Python virtual environment and installs all dependencies.
# Usage: bash scripts/setup.sh [--dev] [--no-torch-cpu-only]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}→ $*${RESET}"; }
success() { echo -e "${GREEN}✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
error()   { echo -e "${RED}✗ $*${RESET}" >&2; exit 1; }

# ─── Parse arguments ─────────────────────────────────────────────────────────
DEV_INSTALL=false
CPU_ONLY_TORCH=true
for arg in "$@"; do
  case "$arg" in
    --dev) DEV_INSTALL=true ;;
    --no-torch-cpu-only) CPU_ONLY_TORCH=false ;;
    --help|-h)
      echo "Usage: bash scripts/setup.sh [--dev] [--no-torch-cpu-only]"
      echo "  --dev              Install development dependencies (pytest, ruff, mypy)"
      echo "  --no-torch-cpu-only  Install torch with CUDA support (larger download)"
      exit 0 ;;
  esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Bharat AI-SoC PS2 — Environment Setup              ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ─── Check Python version ────────────────────────────────────────────────────
PYTHON_CMD=""
for py in python3.12 python3.11 python3; do
  if command -v "$py" &>/dev/null; then
    PY_VERSION=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=${PY_VERSION%%.*}
    PY_MINOR=${PY_VERSION##*.}
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 11 ]; then
      PYTHON_CMD="$py"
      break
    fi
  fi
done

if [ -z "$PYTHON_CMD" ]; then
  error "Python 3.11+ is required. Please install it first:\n  sudo apt-get install python3.11"
fi
success "Using $PYTHON_CMD (version $PY_VERSION)"

# ─── Detect platform ─────────────────────────────────────────────────────────
ARCH=$(uname -m)
OS=$(uname -s)
IS_RPI=false

if [ -f /proc/device-tree/model ]; then
  MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "")
  if echo "$MODEL" | grep -qi "raspberry"; then
    IS_RPI=true
    warn "Raspberry Pi detected ($MODEL)"
    warn "Using CPU-only torch. Inference will be optimised for ARM."
    CPU_ONLY_TORCH=true
  fi
fi

info "Platform: $OS / $ARCH"

# ─── Install system dependencies (Linux only) ────────────────────────────────
if [ "$OS" = "Linux" ]; then
  info "Installing system dependencies..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
      libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
      v4l-utils \
      2>/dev/null || warn "Some system packages could not be installed (may already be present)"
    success "System dependencies installed"
  else
    warn "Non-Debian Linux detected. Install OpenCV system deps manually if needed."
  fi
fi

# ─── Create virtual environment ───────────────────────────────────────────────
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
  warn "Virtual environment already exists at $VENV_DIR. Skipping creation."
else
  info "Creating virtual environment at $VENV_DIR..."
  "$PYTHON_CMD" -m venv "$VENV_DIR"
  success "Virtual environment created"
fi

# ─── Activate venv ───────────────────────────────────────────────────────────
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
info "Activated: $VIRTUAL_ENV"

# ─── Upgrade pip ─────────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --quiet --upgrade pip
success "pip upgraded to $(pip --version | awk '{print $2}')"

# ─── Install PyTorch ─────────────────────────────────────────────────────────
info "Installing PyTorch..."
if $CPU_ONLY_TORCH; then
  pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
  success "PyTorch (CPU-only) installed"
else
  pip install --quiet torch torchvision
  success "PyTorch (with CUDA) installed"
fi

# ─── Install project dependencies ────────────────────────────────────────────
REQ_FILE="version one c2s/module/requirements.txt"
if [ ! -f "$REQ_FILE" ]; then
  error "Requirements file not found: $REQ_FILE"
fi

info "Installing project dependencies from $REQ_FILE..."
pip install --quiet -r "$REQ_FILE"
success "Project dependencies installed"

# ─── Install dev dependencies ────────────────────────────────────────────────
if $DEV_INSTALL; then
  info "Installing development dependencies..."
  pip install --quiet pytest pytest-cov pytest-mock ruff mypy types-requests types-Pillow
  success "Dev dependencies installed"
fi

# ─── Create placeholder directories ─────────────────────────────────────────
info "Creating runtime directories..."
mkdir -p \
  "version one c2s/module/models" \
  "version one c2s/module/database" \
  "version one c2s/module/reports/json" \
  "version one c2s/module/logs" \
  models reports/json logs database

touch \
  "version one c2s/module/reports/.gitkeep" \
  "version one c2s/module/logs/.gitkeep" \
  models/.gitkeep reports/.gitkeep logs/.gitkeep
success "Runtime directories ready"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  Setup complete!${RESET}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════════${RESET}"
echo ""
echo "  Next steps:"
echo "  1. Download the Llama model:"
echo "     ${CYAN}bash scripts/download_models.sh${RESET}"
echo ""
echo "  2. Activate the virtual environment:"
echo "     ${CYAN}source .venv/bin/activate${RESET}"
echo ""
echo "  3. Start the web application:"
echo "     ${CYAN}cd 'version one c2s/module' && python web_app.py${RESET}"
echo "     ${CYAN}→ http://localhost:5000${RESET}"
echo ""
echo "  4. Or start the desktop application:"
echo "     ${CYAN}cd 'version one c2s/module' && python main.py${RESET}"
echo ""
