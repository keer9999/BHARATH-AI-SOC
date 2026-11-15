## 📋 Pull Request Summary

<!-- Provide a brief summary of the changes in this PR -->

### Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [ ] ✨ New feature (non-breaking change that adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to change)
- [ ] 🔧 Refactor (code restructuring without behaviour change)
- [ ] ⚡ Performance improvement
- [ ] 📚 Documentation update
- [ ] 🧪 Test addition / improvement
- [ ] 🔩 CI/CD change

### Linked Issues

<!-- Link the issue(s) this PR addresses -->

Closes #<!-- issue number -->

---

## 🔍 Description of Changes

<!-- Describe what this PR changes and why. Be specific. -->

### What changed:
- 

### Why:
- 

---

## 🧪 Testing

### How was this tested?

- [ ] Unit tests added/updated (`pytest "version one c2s/module/tests/"`)
- [ ] Manual testing on host machine
- [ ] Manual testing on Raspberry Pi 5
- [ ] Manual testing with Pico W sensors connected
- [ ] Manual testing with FPGA (Zynq-7000)
- [ ] No tests required (documentation / config change)

### Test commands run:

```bash
# Paste the commands you ran and their output summary
make test
make lint
```

---

## 📸 Screenshots / Demo

<!-- For UI changes (web app, desktop HUD), include before/after screenshots or a GIF -->

---

## 🏗️ Architecture Impact

<!-- Does this change the system architecture? Update docs/ARCHITECTURE.md if so -->

- [ ] No architecture changes
- [ ] Architecture docs updated (`docs/ARCHITECTURE.md`)
- [ ] API changed — API docs updated (`docs/API_REFERENCE.md`)
- [ ] Hardware setup changed — hardware docs updated (`docs/HARDWARE_SETUP.md`)

---

## ✅ Checklist

**Code quality:**
- [ ] My code follows the project's style guide (passes `make lint`)
- [ ] I have added type hints to all new functions
- [ ] I have written docstrings for all new public functions/classes
- [ ] No debug `print()` statements left in production code

**Testing:**
- [ ] All existing tests pass (`make test`)
- [ ] New tests cover my changes
- [ ] Tests are deterministic and do not require hardware (or marked `@pytest.mark.hardware`)

**Documentation:**
- [ ] I have updated the README if needed
- [ ] I have updated/added docstrings
- [ ] `CHANGELOG.md` updated with a note under `[Unreleased]`

**Review:**
- [ ] I have reviewed my own code before requesting review
- [ ] The PR title follows Conventional Commits format (`feat(scope): description`)
- [ ] The PR is appropriately sized (< 400 lines changed where possible)

