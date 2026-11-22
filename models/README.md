# Neural Model Registry

This directory contains on-device AI model weights and artifacts for the Bharat AI-SoC Waste Perception and Segregation System.

---

## Vision Perception Models

The system incorporates specialized YOLO vision models covering broad material segregation, granular packaging recognition, and targeted specialist categories:

| Model File | Parameter Size | Architecture | Category Count | Functional Role |
| :--- | :--- | :--- | :--- | :--- |
| `best.pt` | ~22.5 MB | YOLOv8s | 5 Classes | Primary Segregation Model: Optimized for fast-path bin sorting on Zynq-7000 (ZedBoard ARM Cortex-A9). |
| `ecosort_waste_yolo11s.pt` | ~19.2 MB | YOLO11s | 66 Classes | Granular Recyclables Model: Provides fine-grained material recognition, feeding semantic context to WasteAdvisor and Llama. |
| `ewaste_a.pt` | ~11.2 MB | YOLOv8n | 12 Classes | Compact Electronic Waste: Specialized on keyboards, batteries, circuit boards, and appliances. |
| `ewaste_b.pt` | ~22.5 MB | YOLOv8s | 77 Classes | Comprehensive Electronics Catalog: Covers consumer appliances, monitors, mobile devices, and medical electronics. |
| `plastic_bag.pt` | ~6.5 MB | YOLOv8n | 1 Class | Flexible Film Specialist: Carrier bags, shrink wrap, and agricultural film. |
| `general_waste.pt` | ~6.5 MB | YOLOv8n | 5 Classes | Municipal Solid Waste: Baseline verification for paper, plastic, metal, glass, and cardboard. |
| `plastic_containers.pt`| ~6.5 MB | YOLOv8n | 3 Classes | Beverage and Food Containers: Plastic bottles, cups, and carrier bags. |
| `resin_codes.pt` | ~6.5 MB | YOLOv8n | 6 Classes | Resin Identification Stamp Reader: Differentiates HDPE (#2), LDPE (#4), PETE (#1), PP (#5), PS (#6), and PVC (#3). |
| `food_organic.pt` | ~11.5 MB | YOLOv8s | 39 Classes | Organic Matter and Textiles: Food waste, agricultural residue, and discarded garments. |
| `bottle_specialist.pt` | ~20.3 MB | YOLOv8s | 1 Class | Single-Use Beverage Container Specialist: Precision detector for plastic and glass beverage containers. |
| `yolov8n.pt` | ~6.5 MB | YOLOv8n | 80 Classes | General COCO Fallback: Identifies everyday objects mapped to waste streams via `COCO_WASTE_MAP`. |

---

## Unified Ensemble Package

For single-file edge distribution, all specialist models can be bundled into a consolidated checkpoint:
- Directory: `models/unified/`
- Target Checkpoint: `models/unified/c2s_waste_ensemble.pt`
- Build Utility: `python scripts/build_unified_model.py`

Refer to [models/unified/README.md](unified/README.md) for loading instructions via PyTorch or ONNX Runtime.

---

## Language Model (Reasoning Modality)

- Model: Meta Llama 3.2 1B Instruct (`llama-3.2-1b-instruct-q4_k_m.gguf`)
- Serialization: GGUF (4-bit quantized `Q4_K_M`)
- Storage Footprint: ~807 MB
- Execution: Evaluates detected visual objects and produces contextual disposal protocols, decomposition estimates, and circular upcycling recommendations.
- Download Script:
  ```bash
  bash scripts/download_models.sh --llama
  ```
  If weights are not present, the system defaults to the deterministic 2,300-rule `WasteAdvisor` knowledge base with zero performance degradation.

---

## Directory Organization and Synchronization

- `models/`: Central staging directory for model checkpoints.
- `src/yolo/weights/`: Default runtime lookup directory utilized by `WasteDetector`.
- `hardware/zynq7000/external/models/`: Storage location staged for external SD card integration on ZedBoard.
