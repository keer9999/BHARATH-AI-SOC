#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/download_llama.sh  —  Download Llama 3.2 1B GGUF for Pi
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PI_DIR/models"
mkdir -p "$MODELS_DIR"

GGUF_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
GGUF_FILE="$MODELS_DIR/llama-3.2-1b-instruct-q4_k_m.gguf"

if [ -f "$GGUF_FILE" ]; then
    echo "✔ Already downloaded: $GGUF_FILE ($(du -sh "$GGUF_FILE" | cut -f1))"
    exit 0
fi

echo "Downloading Llama 3.2 1B Instruct Q4_K_M (~750 MB) ..."
echo "URL: $GGUF_URL"
wget -q --show-progress "$GGUF_URL" -O "$GGUF_FILE"
echo "✔ Saved to: $GGUF_FILE"
echo ""
echo "If you have an ExecuTorch .pte file, place it at:"
echo "  $MODELS_DIR/llama3_2_1b.pte"
echo "And the tokenizer at:"
echo "  $MODELS_DIR/tokenizer.model"
