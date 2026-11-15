#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/run_tests.sh — Run the full test suite
# Usage: bash scripts/run_tests.sh [--cov] [--hardware] [--verbose]
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}→ $*${RESET}"; }
success() { echo -e "${GREEN}✓ $*${RESET}"; }
warn()    { echo -e "${YELLOW}⚠ $*${RESET}"; }
error()   { echo -e "${RED}✗ $*${RESET}" >&2; exit 1; }

# ─── Parse arguments ─────────────────────────────────────────────────────────
COVERAGE=false
INCLUDE_HARDWARE=false
VERBOSE=false
FAST=false

for arg in "$@"; do
  case "$arg" in
    --cov|--coverage) COVERAGE=true ;;
    --hardware)       INCLUDE_HARDWARE=true ;;
    -v|--verbose)     VERBOSE=true ;;
    --fast)           FAST=true ;;
    --help|-h)
      echo "Usage: bash scripts/run_tests.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --cov, --coverage   Generate HTML coverage report"
      echo "  --hardware          Include tests requiring physical hardware"
      echo "  -v, --verbose       Verbose pytest output"
      echo "  --fast              Skip slow tests"
      exit 0 ;;
  esac
done

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║  Bharat AI-SoC PS2 — Test Runner                    ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ─── Activate venv ───────────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  info "Using virtual environment: $VIRTUAL_ENV"
else
  warn "No .venv found — using system Python. Run 'bash scripts/setup.sh --dev' first."
fi

# ─── Verify pytest is available ──────────────────────────────────────────────
if ! command -v pytest &>/dev/null; then
  error "pytest not found. Run: bash scripts/setup.sh --dev"
fi

# ─── Build pytest arguments ───────────────────────────────────────────────────
export PYTHONPATH="src:${PYTHONPATH:-}"

PYTEST_ARGS=(
  "tests/"
  "--tb=short"
  "--color=yes"
)

# Marker filters
MARKERS="not llm"  # LLM tests require the model file
if ! $INCLUDE_HARDWARE; then
  MARKERS="$MARKERS and not hardware"
fi
if $FAST; then
  MARKERS="$MARKERS and not slow"
fi
PYTEST_ARGS+=("-m" "$MARKERS")

# Verbosity
if $VERBOSE; then
  PYTEST_ARGS+=("-v")
else
  PYTEST_ARGS+=("-q")
fi

# Coverage
if $COVERAGE; then
  PYTEST_ARGS+=(
    "--cov=src"
    "--cov-report=html:htmlcov"
    "--cov-report=term-missing"
    "--cov-report=xml"
  )
fi

# ─── Environment for CI-like test run ────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=""
export PICO_SERIAL_PORT="none"

# ─── Run tests ───────────────────────────────────────────────────────────────
info "Running tests..."
echo ""
echo "Command: pytest ${PYTEST_ARGS[*]}"
echo ""

START_TIME=$SECONDS
set +e
pytest "${PYTEST_ARGS[@]}"
EXIT_CODE=$?
set -e
ELAPSED=$((SECONDS - START_TIME))

echo ""
if [ $EXIT_CODE -eq 0 ]; then
  success "All tests passed in ${ELAPSED}s"
  if $COVERAGE; then
    success "Coverage report: htmlcov/index.html"
  fi
else
  error "Tests FAILED (exit code $EXIT_CODE) in ${ELAPSED}s"
fi

exit $EXIT_CODE

