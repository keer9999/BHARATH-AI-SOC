# ─────────────────────────────────────────────────────────────────────────────
# Bharat AI-SoC PS2 — Dockerfile
# EcoSort AI — On-Device Waste Classification & Disposal Assistant
#
# This image is for LOCAL DEVELOPMENT and testing only.
# For edge deployment, use the platform-specific guides:
#   - Raspberry Pi:    raspberry_pi/keerthans_pi/README.md
#   - Zynq-7000:       hardware/zynq7000/README_ZYNQ7000.md
#
# Build:  docker build -t ecosort-ai .
# Run:    docker run -p 5001:5001 ecosort-ai
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

LABEL maintainer="Bharat AI-SoC PS2 Team"
LABEL description="EcoSort AI — On-Device Waste Classification"
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

# ─── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt ./

# Install CPU-only torch (smaller image for local dev testing)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ─── Application code ────────────────────────────────────────────────────────
COPY src/ ./src/

# ─── Runtime directories ─────────────────────────────────────────────────────
RUN mkdir -p /app/src/models /app/src/database /app/src/reports /app/src/logs && \
    chown -R wasteai:wasteai /app

USER wasteai

# ─── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    FLASK_ENV=production \
    ECOSORT_ONNX_ONLY=1 \
    ECOSORT_CUSTOM_MODEL=/app/src/yolo/weights/ecosort_waste_yolo11s.onnx

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/api/status')" || exit 1

CMD ["python", "src/web_app.py", "--host", "0.0.0.0", "--port", "5001"]
