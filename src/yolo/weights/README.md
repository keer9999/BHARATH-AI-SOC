# YOLO Model Weights

This directory contains trained YOLO weights for the waste classification subsystem.

## Files
- `best.pt`: Best fine-tuned checkpoint for multi-class waste segregation. Auto-detected by `WasteDetector`.
- `weights.pt`: Primary baseline weights checkpoint.
- `ecosort_waste_yolo11s.pt`: YOLO11 small architecture weights (optional high-accuracy model).

Weights are loaded by `src/waste_detector.py` according to priority configured in `src/config.py`.
