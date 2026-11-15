# ─────────────────────────────────────────────────────────────────────────────
# Bharat AI-SoC PS2 — Makefile
# Common development commands
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install install-dev run run-desktop test lint format clean \
        docker-build docker-up docker-down setup-pico models

# Default Python and venv paths
PYTHON      := python3
VENV        := .venv
PIP         := $(VENV)/bin/pip
PYTEST      := $(VENV)/bin/pytest
RUFF        := $(VENV)/bin/ruff
MODULE_DIR  := src

# Colour output helpers
BOLD  := \033[1m
RESET := \033[0m
GREEN := \033[32m
CYAN  := \033[36m

## help: Show this help message
help:
	@echo ""
	@echo "$(BOLD)Bharat AI-SoC PS2 — Waste Classification System$(RESET)"
	@echo "$(CYAN)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(RESET)"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /' | \
	  awk -F: '{printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

## install: Create venv and install production dependencies
install:
	@echo "$(BOLD)→ Creating virtual environment...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

## install-dev: Install production + development dependencies
install-dev: install
	$(PIP) install pytest pytest-cov pytest-mock ruff mypy types-requests
	@echo "$(GREEN)✓ Dev dependencies installed$(RESET)"

## run: Start the Flask web application
run:
	@echo "$(BOLD)→ Starting web application at http://localhost:5000$(RESET)"
	$(VENV)/bin/python "$(MODULE_DIR)/web_app.py"

## run-desktop: Start the OpenCV desktop application
run-desktop:
	@echo "$(BOLD)→ Starting desktop application...$(RESET)"
	$(VENV)/bin/python "$(MODULE_DIR)/main.py"

## test: Run the full test suite
test:
	@echo "$(BOLD)→ Running tests...$(RESET)"
	PYTHONPATH=$(MODULE_DIR) $(PYTEST) tests/ \
	  -v --tb=short
	@echo "$(GREEN)✓ Tests complete$(RESET)"

## test-cov: Run tests with HTML coverage report
test-cov:
	PYTHONPATH=$(MODULE_DIR) $(PYTEST) tests/ \
	  --cov="$(MODULE_DIR)" \
	  --cov-report=html:htmlcov \
	  --cov-report=term-missing \
	  -v

## lint: Run ruff linter (check only)
lint:
	@echo "$(BOLD)→ Running ruff linter...$(RESET)"
	$(RUFF) check "$(MODULE_DIR)" tests/ --output-format=github
	@echo "$(GREEN)✓ Lint passed$(RESET)"

## format: Auto-format code with ruff
format:
	@echo "$(BOLD)→ Formatting code...$(RESET)"
	$(RUFF) format "$(MODULE_DIR)" tests/
	$(RUFF) check --fix "$(MODULE_DIR)" tests/
	@echo "$(GREEN)✓ Code formatted$(RESET)"

## models: Download AI models (Llama GGUF)
models:
	bash scripts/download_models.sh

## docker-build: Build the Docker image
docker-build:
	@echo "$(BOLD)→ Building Docker image...$(RESET)"
	docker build -t bharat-aisoc-waste:latest .
	@echo "$(GREEN)✓ Docker image built$(RESET)"

## docker-up: Start Docker Compose stack
docker-up:
	docker compose up --build -d
	@echo "$(GREEN)✓ Stack started. Web UI: http://localhost:5000$(RESET)"

## docker-down: Stop Docker Compose stack
docker-down:
	docker compose down
	@echo "$(GREEN)✓ Stack stopped$(RESET)"

## docker-logs: Follow Docker Compose logs
docker-logs:
	docker compose logs -f waste-web

## setup-pico: Flash MicroPython firmware guide
setup-pico:
	@echo "$(BOLD)Raspberry Pi Pico W Setup$(RESET)"
	@echo "1. Hold BOOTSEL on Pico W while connecting USB"
	@echo "2. Download MicroPython firmware: https://micropython.org/download/RPI_PICO_W/"
	@echo "3. Copy .uf2 file to the RPI-RP2 drive"
	@echo "4. Flash sensor firmware:"
	@echo "   cd '$(MODULE_DIR)/pico_sensors'"
	@echo "   # Use Thonny or mpremote to upload main.py"

## clean: Remove build artifacts and caches
clean:
	@echo "$(BOLD)→ Cleaning build artifacts...$(RESET)"
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned$(RESET)"

## clean-all: Remove venv, Docker images, and all caches
clean-all: clean
	rm -rf $(VENV)
	docker compose down --volumes --rmi local 2>/dev/null || true
	@echo "$(GREEN)✓ Full clean complete$(RESET)"

