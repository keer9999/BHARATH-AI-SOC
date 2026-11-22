# ─────────────────────────────────────────────────────────────────────────────
# Bharat AI-SoC PS2 — Dockerfile
# AI-Powered Waste Classification & Disposal Assistant
# Base: python:3.11-slim (for portability; for RPi use python:3.11-slim-bookworm)
#
# Render free tier (512 MB RAM):
#   - Uses unified 66-class ONNX model only (ecosort_waste_yolo11s.onnx)
#   - Skips 8 specialist .pt models via ECOSORT_ONNX_ONLY=1
#   - Installs onnxruntime-cpu instead of full torch (~250 MB saving)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

LABEL maintainer="Bharat AI-SoC PS2 Team"
LABEL description="AI-Powered Waste Classification & Disposal Assistant"
LABEL version="2.0.0"

# ─── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ─── Create non-root user ────────────────────────────────────────────────────
RUN groupadd -r wasteai && useradd -r -g wasteai -d /app -s /bin/bash wasteai

WORKDIR /app

# ─── Python dependencies (slim — no heavy torch for cloud deploy) ─────────────
COPY requirements.txt ./requirements.txt
COPY requirements-render.txt ./requirements-render.txt

# Install the render-specific slim requirements (no torch, uses onnxruntime)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-render.txt

# ─── Application code ────────────────────────────────────────────────────────
COPY src/ ./src/

# ─── Copy only the unified ONNX model (37 MB, covers all 66 classes) ─────────
# The 8 specialist .pt files are NOT copied into the image to stay under 512 MB.
RUN mkdir -p /app/src/yolo/weights \
    /app/src/models \
    /app/src/database \
    /app/src/reports \
    /app/src/logs

# Note: ONNX file must be tracked in git (not in .gitignore).
# It is copied from src/ above automatically.

RUN chown -R wasteai:wasteai /app

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
    WASTE_LOG_PATH=/app/src/logs/waste_system.log \
    ECOSORT_ONNX_ONLY=1 \
    ECOSORT_CUSTOM_MODEL=/app/src/yolo/weights/ecosort_waste_yolo11s.onnx \
    ENABLE_SAHI=false

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/status')" || exit 1

CMD ["python", "src/web_app.py", "--host", "0.0.0.0", "--port", "5000"]
