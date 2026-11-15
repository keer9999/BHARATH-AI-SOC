# Contributing to Bharat AI-SoC PS2 — Waste Classification System

Thank you for your interest in contributing! This project is part of the Bharat AI-SoC Challenge 2026-27 and we welcome all contributions, from bug reports to new features.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Commit Message Convention](#commit-message-convention)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Testing](#testing)

---

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## Getting Started

### 1. Fork & Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/bharat-aisoc-ps2.git
cd bharat-aisoc-ps2
git remote add upstream https://github.com/keer9999/bharath-ai-soc.git
```

### 2. Set Up Development Environment

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

### 3. Create a Feature Branch

```bash
git checkout -b feat/your-descriptive-feature-name
# or
git checkout -b fix/brief-description-of-bug
```

---

## Development Workflow

```
fork → clone → branch → develop → test → lint → commit → push → PR
```

1. **Sync** with upstream before starting new work:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
2. **Develop** your changes in small, focused commits.
3. **Test** before committing:
   ```bash
   make test
   ```
4. **Lint** your code:
   ```bash
   make lint
   ```
5. **Push** to your fork and open a Pull Request.

---

## Code Style

We use **Ruff** for linting and formatting (configured in `pyproject.toml`).

```bash
# Auto-format
ruff format .

# Lint
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Key Guidelines

- **Line length**: 100 characters maximum
- **Quotes**: Double quotes for strings
- **Type hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings for all public functions and classes
- **Imports**: Sorted with `ruff` (isort-compatible)

### Example

```python
def classify_waste(frame: np.ndarray, confidence_threshold: float = 0.5) -> list[Detection]:
    """Classify waste objects in an image frame.

    Args:
        frame: BGR image frame from OpenCV (H x W x 3).
        confidence_threshold: Minimum confidence score to include a detection.

    Returns:
        List of Detection objects with class, confidence, and bounding box.

    Raises:
        ValueError: If frame is not a valid BGR image.
    """
    ...
```

---

## Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format

```
<type>(<scope>): <short description>

[optional body]

[optional footer(s)]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Code style changes (formatting, etc.) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | A code change that improves performance |
| `test` | Adding or correcting tests |
| `ci` | Changes to CI configuration files |
| `chore` | Other changes that don't modify src or test files |
| `revert` | Reverts a previous commit |

### Scopes

`detector`, `llm`, `sensors`, `web`, `desktop`, `fpga`, `db`, `reports`, `alerts`, `config`, `docs`, `ci`

### Examples

```
feat(detector): add YOLOv8s model support with auto-selection
fix(sensors): handle DHT11 timeout gracefully with retry logic
docs(api): document /api/chat endpoint with request/response schema
perf(llm): reduce Llama inference latency by 30% with batching
test(detector): add unit tests for 5-class confidence thresholding
ci: add Python 3.12 to test matrix
```

---

## Pull Request Process

1. **Ensure all tests pass** (`make test`) and linting is clean (`make lint`).
2. **Update documentation** if you've changed any public APIs or behavior.
3. **Write or update tests** to cover your changes.
4. **Fill out the PR template** completely when opening the PR.
5. **Request a review** from a maintainer.
6. **Address review comments** promptly.
7. **Squash commits** before merge if requested.

### PR Title Format

Follow the same Conventional Commits format:
```
feat(web): add real-time waste composition pie chart to dashboard
```

---

## Issue Reporting

Use the GitHub Issue templates:

- 🐛 **[Bug Report](.github/ISSUE_TEMPLATE/bug_report.md)** — Something isn't working
- ✨ **[Feature Request](.github/ISSUE_TEMPLATE/feature_request.md)** — Suggest a new feature

### Before Opening an Issue

1. Search existing issues to avoid duplicates.
2. Try the latest version on `main`.
3. Collect relevant logs from `logs/waste_system.log`.

---

## Testing

```bash
# Run all tests
make test

# Run specific test file
pytest "version one c2s/module/tests/test_material_mapping.py" -v

# Run with coverage report
pytest --cov="version one c2s/module" --cov-report=html
```

### Test Structure

- Unit tests in `version one c2s/module/tests/`
- Tests must be deterministic and not require hardware
- Mock external dependencies (camera, serial port, LLM) in unit tests
- Use `pytest.mark.hardware` for tests requiring physical devices

---

## Questions?

Open a [Discussion](https://github.com/keer9999/bharath-ai-soc/discussions) or reach out via the issue tracker.

**Happy Contributing! 🇮🇳**

