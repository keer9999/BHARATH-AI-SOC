#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-click environment setup for C2S Bharat AI-SoC
# Sets up Python venv, installs deps, downloads models, and verifies the system.
#
# Usage:
#   bash setup.sh           # Full setup (laptop)
#   bash setup.sh --zynq    # Zynq-7000 / ZedBoard headless setup
#   bash setup.sh --test    # Setup + run self-test
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}→ $*${RESET}"; }
success() { echo -e "${GREEN}✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
error()   { echo -e "${RED}✗ $*${RESET}" >&2; exit 1; }

ZYNQ_MODE=false
RUN_TEST=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zynq) ZYNQ_MODE=true; shift ;;
    --test) RUN_TEST=true; shift ;;
    --help|-h)
      echo "Usage: bash setup.sh [--zynq] [--test]"
      exit 0 ;;
    *) shift ;;
  esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  C2S — Bharat AI-SoC Waste Detection — Setup Script     ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
info "Checking Python version..."
PYTHON_CMD=""
for py in python3.12 python3.11 python3.10 python3; do
  if command -v "$py" &>/dev/null; then
    VER=$($py --version 2>&1 | grep -oP '3\.\d+')
    MAJOR=3
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [ "$MINOR" -ge 10 ]; then
      PYTHON_CMD="$py"
      success "Found $py ($VER)"
      break
    fi
  fi
done
[ -z "$PYTHON_CMD" ] && error "Python 3.10+ is required. Install it first."

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ ! -f ".venv/bin/activate" ]; then
  info "Creating virtual environment (.venv)..."
  $PYTHON_CMD -m venv .venv
  success "Virtual environment created"
else
  success "Virtual environment already exists"
fi

# Activate
source .venv/bin/activate
info "Using: $(which python) ($(python --version))"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
info "Upgrading pip..."
pip install --quiet --upgrade pip

# ── 4. Install dependencies ───────────────────────────────────────────────────
info "Installing Python dependencies from requirements.txt..."
if [ "$ZYNQ_MODE" = true ]; then
  # Zynq: skip llama-cpp-python (compiled separately), no streamlit needed
  pip install --quiet ultralytics opencv-python-headless numpy flask \
      Pillow pyserial requests reportlab matplotlib pandas
  warn "Zynq mode: skipping llama-cpp-python (install separately on board)"
else
  pip install --quiet -r requirements.txt || {
    warn "Full requirements install failed — trying core packages..."
    pip install ultralytics opencv-python numpy flask Pillow pyserial \
        requests reportlab matplotlib pandas plotly streamlit
  }
fi
success "Dependencies installed"

# ── 5. Download models ────────────────────────────────────────────────────────
info "Checking / downloading models..."
bash scripts/download_models.sh --yolo || warn "YOLO model sync skipped (run manually if needed)"

# ── 6. Create missing directories ─────────────────────────────────────────────
info "Creating runtime directories..."
mkdir -p logs database evidence reports/json src/logs src/database

# ── 7. Verify imports ─────────────────────────────────────────────────────────
info "Verifying Python imports..."
python -c "
import sys
modules = ['ultralytics', 'cv2', 'flask', 'numpy', 'PIL', 'serial']
failed = []
for m in modules:
    try:
        __import__(m)
        print(f'  ✓ {m}')
    except ImportError as e:
        print(f'  ✗ {m}: {e}')
        failed.append(m)
if failed:
    print(f'WARNING: {len(failed)} module(s) missing: {failed}')
    sys.exit(1)
else:
    print('All core imports OK')
" || warn "Some imports failed — check requirements above"

# ── 8. Run self-test ──────────────────────────────────────────────────────────
if [ "$RUN_TEST" = true ]; then
  info "Running self-test (10 frames, no hardware)..."
  python src/main.py --test --headless
  success "Self-test passed!"
fi

echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════${RESET}"
success "Setup complete!"
echo ""
echo "  Run the system:"
echo "    source .venv/bin/activate"
echo "    python src/main.py --test --headless   # Self-test"
echo "    python src/main.py --headless           # Webcam mode"
echo "    python src/web_app.py                   # Web dashboard"
echo ""
if [ "$ZYNQ_MODE" = false ]; then
  echo "  For Zynq-7000:"
  echo "    bash setup.sh --zynq"
  echo "    python hardware/zynq7000/bram/main_zynq.py --test"
fi
echo ""
