# ─────────────────────────────────────────────────────────────────────────────
# Bharat AI-SoC PS2 — Dockerfile
# AI-Powered Waste Classification & Disposal Assistant
# Base: python:3.11-slim (for portability; for RPi use python:3.11-slim-bookworm)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

LABEL maintainer="Bharat AI-SoC PS2 Team"
LABEL description="AI-Powered Waste Classification & Disposal Assistant"
LABEL version="1.0.0"

# ─── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime dependencies
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    # Camera support
    v4l-utils \
    # Build tools for llama-cpp-python
    build-essential \
    cmake \
    git \
    # Serial port support
    udev \
    && rm -rf /var/lib/apt/lists/*

# ─── Create non-root user ────────────────────────────────────────────────────
RUN groupadd -r wasteai && useradd -r -g wasteai -d /app -s /bin/bash wasteai

# ─── Working directory ───────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies ─────────────────────────────────────────────────────
# Copy only requirements first for better layer caching
COPY requirements.txt ./requirements.txt

# Install Python dependencies (CPU-only torch for smaller image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# ─── Application code ────────────────────────────────────────────────────────
COPY src/ ./src/

# ─── Persistent volume mount points ──────────────────────────────────────────
# Models, database, reports, and logs are volume-mounted at runtime
RUN mkdir -p /app/src/models \
    /app/src/database \
    /app/src/reports \
    /app/src/logs \
    /app/src/yolo/weights && \
    ([ ! -f /app/src/yolo/weights/weights.pt ] && [ -f /app/yolov8n.pt ] && cp /app/yolov8n.pt /app/src/yolo/weights/weights.pt || true) && \
    chown -R wasteai:wasteai /app

# ─── Switch to non-root user ─────────────────────────────────────────────────
USER wasteai

# ─── Environment variables ───────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    FLASK_APP=src/web_app.py \
    FLASK_ENV=production \
    WASTE_MODEL_PATH=/app/src/models \
    WASTE_DB_PATH=/app/src/database/waste_data.db \
    WASTE_REPORTS_PATH=/app/src/reports \
    WASTE_LOG_PATH=/app/src/logs/waste_system.log

# ─── Expose Flask port ───────────────────────────────────────────────────────
EXPOSE 5000

# ─── Health check ────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/status')" || exit 1

# ─── Entrypoint ──────────────────────────────────────────────────────────────
CMD ["python", "src/web_app.py", "--host", "0.0.0.0", "--port", "5000"]
