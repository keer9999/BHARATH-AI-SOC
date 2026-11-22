#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/start_pi.sh  —  Start EcoSort AI on Keerthan's Raspberry Pi
#
# Usage:
#   bash raspberry_pi/keerthans_pi/scripts/start_pi.sh
#   bash raspberry_pi/keerthans_pi/scripts/start_pi.sh --port 5001
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PI_DIR/../.." && pwd)"
PORT="${1:-5001}"

# Activate virtual environment
source "$PI_DIR/.venv/bin/activate"

# Load Pi-specific env vars
source "$PI_DIR/config/pi_config.env"

# Override PORT if passed as arg
export PORT="$PORT"

echo "═══════════════════════════════════════════════════"
echo "  EcoSort AI — Starting on Keerthan's Pi"
echo "  ONNX model : $ECOSORT_CUSTOM_MODEL"
echo "  Llama model: $LLAMA_GGUF_PATH"
echo "  Port       : $PORT"
echo "═══════════════════════════════════════════════════"

# Create runtime dirs
mkdir -p "$PI_DIR/database" "$PI_DIR/reports" "$PI_DIR/logs"

cd "$REPO_ROOT/src"
exec python web_app.py --host 0.0.0.0 --port "$PORT"
