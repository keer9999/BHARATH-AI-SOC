#!/bin/bash
# ==============================================================================
# ZedBoard (Zynq-7000) Automated Deployment & Setup Script
# AI-Enabled Smart Waste Segregation & Monitoring System
#
# Target: Avnet / Digilent ZedBoard (XC7Z020)
# Memory: 512 MB Physical DDR3 + 8 GB External Storage (/mnt/ext_storage)
# ==============================================================================
# Usage:
#   # Direct network deployment (SSH):
#   ./deploy_to_zedboard.sh [OPTIONS] [ZEDBOARD_IP]
#
#   # Local SD card staging (offline, no network required):
#   ./deploy_to_zedboard.sh --stage /media/$USER/BOOT_OR_ROOTFS
#
#   # Dry run verification:
#   ./deploy_to_zedboard.sh --dry-run [ZEDBOARD_IP]
#
# Options:
#   --ip <IP>           ZedBoard IP address (default: 192.168.1.120)
#   --user <USER>       Remote SSH user (default: root)
#   --port <PORT>       SSH port (default: 22)
#   --key <PATH>        Path to SSH private key
#   --stage <DIR>       Stage files locally into target directory instead of SSH
#   --dry-run           Inspect actions without transferring files
#   --check             Run pre-flight connectivity check and exit
#   -h, --help          Show this help message
# ==============================================================================

set -euo pipefail

ZED_IP="192.168.1.120"
ZED_USER="root"
ZED_PORT="22"
SSH_KEY=""
STAGE_DIR=""
DRY_RUN=false
CHECK_ONLY=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}/.."
REPO_ROOT="${SCRIPT_DIR}/../../.."

# ── Argument Parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)
            ZED_IP="$2"
            shift 2
            ;;
        --user)
            ZED_USER="$2"
            shift 2
            ;;
        --port)
            ZED_PORT="$2"
            shift 2
            ;;
        --key)
            SSH_KEY="$2"
            shift 2
            ;;
        --stage)
            STAGE_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        -h|--help)
            sed -n '2,26p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            if [[ ! "$1" =~ ^- ]]; then
                ZED_IP="$1"
                shift
            else
                echo "Unknown option: $1"
                exit 1
            fi
            ;;
    esac
done

SSH_OPTS=(-p "${ZED_PORT}" -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new)
if [ -n "${SSH_KEY}" ]; then
    SSH_OPTS+=(-i "${SSH_KEY}")
fi

echo "=================================================================="
echo "  ZedBoard Deployment — AI-Enabled Waste System"
echo "  Target IP:   ${ZED_IP}"
echo "  Target User: ${ZED_USER}"
if [ -n "${STAGE_DIR}" ]; then
    echo "  Staging Dir: ${STAGE_DIR} (Local offline mode)"
fi
if [ "$DRY_RUN" = true ]; then
    echo "  Mode:        DRY RUN (no modifications)"
fi
echo "=================================================================="

# ── Locate Model Files ────────────────────────────────────────────────────────
YOLO_MODEL=""
BEST_MODEL=""

for candidate in \
    "${BASE_DIR}/external/models/yolov8n.pt" \
    "${REPO_ROOT}/models/yolov8n.pt" \
    "${REPO_ROOT}/src/yolov8n.pt"; do
    if [ -f "$candidate" ]; then
        YOLO_MODEL="$candidate"
        break
    fi
done

for candidate in \
    "${BASE_DIR}/external/models/best.pt" \
    "${REPO_ROOT}/models/best.pt" \
    "${REPO_ROOT}/src/yolo/weights/best.pt"; do
    if [ -f "$candidate" ]; then
        BEST_MODEL="$candidate"
        break
    fi
done

echo "Model Resolution:"
echo "  yolov8n: ${YOLO_MODEL:-NOT FOUND (will use Ultralytics auto-download)}"
echo "  best.pt: ${BEST_MODEL:-NOT FOUND}"

# ── Offline / Staging Mode ───────────────────────────────────────────────────
if [ -n "${STAGE_DIR}" ]; then
    echo ""
    echo "[Staging] Exporting payload to local directory: ${STAGE_DIR}"
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "${STAGE_DIR}/opt/c2s"
        mkdir -p "${STAGE_DIR}/mnt/ext_storage/models"
        mkdir -p "${STAGE_DIR}/mnt/ext_storage/database"
        mkdir -p "${STAGE_DIR}/mnt/ext_storage/logs"
        mkdir -p "${STAGE_DIR}/mnt/ext_storage/reports"
        mkdir -p "${STAGE_DIR}/mnt/ext_storage/web"

        cp -v "${BASE_DIR}/bram/"*.py "${STAGE_DIR}/opt/c2s/"
        cp -v "${BASE_DIR}/bram/requirements.txt" "${STAGE_DIR}/opt/c2s/"
        cp -v "${BASE_DIR}/external/"*.py "${STAGE_DIR}/mnt/ext_storage/"
        cp -rv "${BASE_DIR}/external/web/"* "${STAGE_DIR}/mnt/ext_storage/web/"
        if [ -d "${BASE_DIR}/external/pico_sensors" ]; then
            cp -rv "${BASE_DIR}/external/pico_sensors" "${STAGE_DIR}/mnt/ext_storage/"
        fi
        if [ -n "$YOLO_MODEL" ]; then
            cp -v "$YOLO_MODEL" "${STAGE_DIR}/mnt/ext_storage/models/yolov8n.pt"
        fi
        if [ -n "$BEST_MODEL" ]; then
            cp -v "$BEST_MODEL" "${STAGE_DIR}/mnt/ext_storage/models/best.pt"
        fi
    fi
    echo "Staging complete! Copy files to SD card partitions:"
    echo "  Partition 1 (Rootfs): /opt/c2s"
    echo "  Partition 2 (Ext4/SD): /mnt/ext_storage"
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed successfully. Model paths and arguments validated."
    exit 0
fi

# ── Pre-flight Check ──────────────────────────────────────────────────────────
echo ""
echo "[Pre-flight] Testing SSH connectivity to ${ZED_USER}@${ZED_IP}:${ZED_PORT}..."
if ! ssh "${SSH_OPTS[@]}" "${ZED_USER}@${ZED_IP}" "echo 'SSH connection verified'" 2>/dev/null; then
    echo "WARNING: Cannot connect to ${ZED_USER}@${ZED_IP} via SSH."
    echo "Please check:"
    echo "  1. ZedBoard is powered on and connected to Ethernet"
    echo "  2. IP address matches your ZedBoard (${ZED_IP})"
    echo "  3. SSH server is running on target (port ${ZED_PORT})"
    echo "  Tip: Use --stage <DIR> to provision SD card offline without SSH."
    if [ "$CHECK_ONLY" = true ]; then
        exit 1
    fi
    read -r -p "Continue anyway? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY]) ;;
        *) exit 1 ;;
    esac
else
    echo "Pre-flight check passed."
    if [ "$CHECK_ONLY" = true ]; then
        exit 0
    fi
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry run completed successfully. All pre-flight parameters verified."
    exit 0
fi

# ── 1. Create Directories on ZedBoard ─────────────────────────────────────────
echo "[1/6] Preparing filesystems on ZedBoard..."
ssh "${SSH_OPTS[@]}" "${ZED_USER}@${ZED_IP}" "
    mkdir -p /opt/c2s
    mkdir -p /mnt/ext_storage/models
    mkdir -p /mnt/ext_storage/database
    mkdir -p /mnt/ext_storage/logs
    mkdir -p /mnt/ext_storage/reports
    mkdir -p /mnt/ext_storage/web
    ln -sfn /mnt/ext_storage /opt/c2s/external
"

# ── 2. Configure Swap Space on External Memory ────────────────────────────────
echo "[2/6] Configuring virtual swap on external storage..."
ssh "${SSH_OPTS[@]}" "${ZED_USER}@${ZED_IP}" "
    if [ -d /mnt/ext_storage ] && [ ! -f /mnt/ext_storage/swapfile ]; then
        echo 'Creating 2GB swapfile on external memory...'
        dd if=/dev/zero of=/mnt/ext_storage/swapfile bs=1M count=2048 2>/dev/null || true
        chmod 600 /mnt/ext_storage/swapfile || true
        mkswap /mnt/ext_storage/swapfile || true
    fi
    if [ -f /mnt/ext_storage/swapfile ]; then
        swapon /mnt/ext_storage/swapfile 2>/dev/null || true
    fi
    free -h || cat /proc/meminfo | grep -E 'MemTotal|MemFree|SwapTotal'
"

# ── 3. Upload BRAM / Low-Footprint Control Files ──────────────────────────────
echo "[3/6] Deploying BRAM / OCM modules to /opt/c2s/ ..."
scp "${SSH_OPTS[@]}" "${BASE_DIR}/bram/"*.py "${ZED_USER}@${ZED_IP}:/opt/c2s/"
scp "${SSH_OPTS[@]}" "${BASE_DIR}/bram/requirements.txt" "${ZED_USER}@${ZED_IP}:/opt/c2s/"

# ── 4. Upload External Memory Files (Large Modules & Web) ────────────────────
echo "[4/6] Deploying application modules to External Storage..."
scp "${SSH_OPTS[@]}" "${BASE_DIR}/external/"*.py "${ZED_USER}@${ZED_IP}:/mnt/ext_storage/"
scp -r "${SSH_OPTS[@]}" "${BASE_DIR}/external/web/"* "${ZED_USER}@${ZED_IP}:/mnt/ext_storage/web/"

if [ -d "${BASE_DIR}/external/pico_sensors" ]; then
    scp -r "${SSH_OPTS[@]}" "${BASE_DIR}/external/pico_sensors" "${ZED_USER}@${ZED_IP}:/mnt/ext_storage/"
fi

# ── 5. Upload AI Model Weights to External Memory ────────────────────────────
echo "[5/6] Uploading YOLO models to /mnt/ext_storage/models..."
if [ -n "$YOLO_MODEL" ]; then
    scp "${SSH_OPTS[@]}" "$YOLO_MODEL" "${ZED_USER}@${ZED_IP}:/mnt/ext_storage/models/yolov8n.pt"
fi
if [ -n "$BEST_MODEL" ]; then
    scp "${SSH_OPTS[@]}" "$BEST_MODEL" "${ZED_USER}@${ZED_IP}:/mnt/ext_storage/models/best.pt"
fi

# ── 6. Setup Auto-Start Service (Supports systemd or init.d) ──────────────────
echo "[6/6] Configuring startup services..."
ssh "${SSH_OPTS[@]}" "${ZED_USER}@${ZED_IP}" "
    if command -v systemctl >/dev/null 2>&1; then
        echo 'Configuring systemd service...'
    else
        echo 'systemd not found; installing standalone startup script in /opt/c2s/start.sh'
        cat << 'EOF' > /opt/c2s/start.sh
#!/bin/sh
cd /opt/c2s && python3 main_zynq.py
EOF
        chmod +x /opt/c2s/start.sh
    fi
"
if [ -f "${SCRIPT_DIR}/c2s-waste-zedboard.service" ]; then
    scp "${SSH_OPTS[@]}" "${SCRIPT_DIR}/c2s-waste-zedboard.service" "${ZED_USER}@${ZED_IP}:/etc/systemd/system/c2s-waste.service" 2>/dev/null || true
    ssh "${SSH_OPTS[@]}" "${ZED_USER}@${ZED_IP}" "
        if command -v systemctl >/dev/null 2>&1; then
            systemctl daemon-reload 2>/dev/null || true
            systemctl enable c2s-waste.service 2>/dev/null || true
        fi
    " 2>/dev/null || true
fi

echo "=================================================================="
echo "  DEPLOYMENT TO ZEDBOARD COMPLETE!"
echo "=================================================================="
echo "To start the system interactively:"
echo "  ssh ${ZED_USER}@${ZED_IP} -p ${ZED_PORT}"
echo "  cd /opt/c2s"
echo "  python3 main_zynq.py"
echo ""
echo "To start the Web Dashboard:"
echo "  cd /mnt/ext_storage && python3 -m http.server 5000"
echo "  or via Flask: python3 /mnt/ext_storage/web_app.py"
echo "=================================================================="
