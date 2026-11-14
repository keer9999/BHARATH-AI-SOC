#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/download_models.sh — Download AI model files
# Downloads the Llama 3.2 1B GGUF model from Hugging Face.
# Usage: bash scripts/download_models.sh [--model <name>]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}→ $*${RESET}"; }
success() { echo -e "${GREEN}✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
error()   { echo -e "${RED}✗ $*${RESET}" >&2; exit 1; }

# ─── Default model configuration ─────────────────────────────────────────────
MODEL_NAME="llama-3.2-1b-instruct-q4_k_m.gguf"
MODEL_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
MODEL_DIR="version one c2s/module/models"

# Allow override via argument
for arg in "$@"; do
  case "$arg" in
    --model) MODEL_NAME="$2"; shift ;;
    --dir) MODEL_DIR="$2"; shift ;;
    --help|-h)
      echo "Usage: bash scripts/download_models.sh [--model <filename>] [--dir <path>]"
      echo ""
      echo "Available GGUF variants (from bartowski/Llama-3.2-1B-Instruct-GGUF):"
      echo "  llama-3.2-1b-instruct-q4_k_m.gguf  (~800 MB)  ← Recommended"
      echo "  llama-3.2-1b-instruct-q5_k_m.gguf  (~1.0 GB)  Higher quality"
      echo "  llama-3.2-1b-instruct-q8_0.gguf    (~1.3 GB)  Best quality"
      echo "  llama-3.2-1b-instruct-q2_k.gguf    (~550 MB)  Smallest/fastest"
      exit 0 ;;
  esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Bharat AI-SoC PS2 — Model Downloader               ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ─── Check destination ───────────────────────────────────────────────────────
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

if [ -f "$MODEL_PATH" ]; then
  success "Model already exists at $MODEL_PATH"
  echo "  To re-download, delete the file first."
  exit 0
fi

mkdir -p "$MODEL_DIR"
info "Model will be saved to: $MODEL_PATH"

# ─── Check download tool ──────────────────────────────────────────────────────
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

# ─── Download model ───────────────────────────────────────────────────────────
HF_URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_NAME}"

warn "Downloading Llama 3.2 1B GGUF (may take several minutes on slow connections)"
info "Source: $HF_URL"
echo ""

case "$DOWNLOAD_CMD" in
  hf)
    info "Using huggingface-cli (fastest, supports resuming)..."
    huggingface-cli download "$MODEL_REPO" "$MODEL_NAME" --local-dir "$MODEL_DIR"
    ;;
  wget)
    info "Using wget..."
    wget --show-progress -c -O "$MODEL_PATH" "$HF_URL"
    ;;
  curl)
    info "Using curl..."
    curl -L --progress-bar -o "$MODEL_PATH" "$HF_URL"
    ;;
esac

# ─── Verify download ─────────────────────────────────────────────────────────
if [ -f "$MODEL_PATH" ]; then
  SIZE_MB=$(($(stat -c%s "$MODEL_PATH") / 1024 / 1024))
  if [ "$SIZE_MB" -lt 100 ]; then
    warn "Downloaded file seems small (${SIZE_MB} MB). Verify the model loaded correctly."
  else
    success "Model downloaded successfully: $MODEL_PATH (${SIZE_MB} MB)"
  fi
else
  error "Download failed — file not found at $MODEL_PATH"
fi

# ─── Quick verification ───────────────────────────────────────────────────────
info "Verifying model loads correctly..."
if source .venv/bin/activate 2>/dev/null; then
  python3 -c "
from llama_cpp import Llama
print('Loading model...')
llm = Llama(model_path='$MODEL_PATH', n_ctx=512, n_threads=2, verbose=False)
out = llm('Hello', max_tokens=10, echo=False)
print('✓ Model loaded and inference works!')
del llm
" && success "Model verified!" || warn "Could not verify model (llama-cpp-python may not be installed yet — run setup.sh first)"
else
  warn "Virtual environment not found — skipping model verification. Run setup.sh first."
fi

echo ""
echo -e "${BOLD}${GREEN}Model ready!${RESET}"
echo "  Path: $MODEL_PATH"
echo "  Next: source .venv/bin/activate && python 'version one c2s/module/web_app.py'"
echo ""
